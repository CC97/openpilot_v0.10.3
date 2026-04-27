#!/usr/bin/env python3
"""
Convert openpilot v0.10.3 driving_vision.onnx to a PyTorch module and compare
ONNX Runtime vs PyTorch outputs on a video.

Required packages:
  pip install torch onnx onnxruntime onnx2torch opencv-python numpy

The script saves the converted PyTorch module with torch.save(...). This is
intended for local experiments, such as gradient-based attacks against the
vision model's lead output.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ONNX_PATH = REPO_ROOT / "selfdrive/modeld/models/driving_vision.onnx"
DEFAULT_VIDEO_PATH = REPO_ROOT / "tools/sim/bridge/video.hevc"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "driving_vision_torch.pt"

VISION_OUTPUT_SIZE = 1576
LEAD_SLICE = slice(917, 1061)
LEAD_MU_SIZE = 3 * 6 * 4


@dataclass
class OutputError:
  count: int = 0
  sum_abs: float = 0.0
  sum_sq: float = 0.0
  max_abs: float = 0.0

  def update(self, diff: Any) -> None:
    import numpy as np

    diff = np.asarray(diff, dtype=np.float64)
    abs_diff = np.abs(diff)
    self.count += diff.size
    self.sum_abs += float(abs_diff.sum())
    self.sum_sq += float(np.square(diff).sum())
    self.max_abs = max(self.max_abs, float(abs_diff.max(initial=0.0)))

  @property
  def mae(self) -> float:
    return self.sum_abs / self.count if self.count else math.nan

  @property
  def rmse(self) -> float:
    return math.sqrt(self.sum_sq / self.count) if self.count else math.nan


def require_modules() -> None:
  missing = []
  for module_name in ("numpy", "cv2", "onnx", "onnxruntime", "torch", "onnx2torch"):
    try:
      importlib.import_module(module_name)
    except ImportError:
      missing.append(module_name)

  if missing:
    packages = {
      "cv2": "opencv-python",
      "onnx2torch": "onnx2torch",
    }
    install_names = [packages.get(name, name) for name in missing]
    raise SystemExit(
      "Missing required Python modules: "
      + ", ".join(missing)
      + "\nInstall them, for example:\n  python3 -m pip install "
      + " ".join(install_names)
    )


def parse_image(frame: Any) -> Any:
  import cv2
  import numpy as np

  frame = cv2.resize(frame, (512, 256))
  yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)

  h = (yuv.shape[0] * 2) // 3
  w = yuv.shape[1]
  parsed = np.zeros((6, h // 2, w // 2), dtype=np.float32)

  parsed[0] = yuv[0:h:2, 0::2]
  parsed[1] = yuv[1:h:2, 0::2]
  parsed[2] = yuv[0:h:2, 1::2]
  parsed[3] = yuv[1:h:2, 1::2]
  parsed[4] = yuv[h:h + h // 4].reshape((h // 2, w // 2))
  parsed[5] = yuv[h + h // 4:h + h // 2].reshape((h // 2, w // 2))
  return parsed


def make_model_input(prev_frame: Any, cur_frame: Any) -> Any:
  import numpy as np

  prev_img = parse_image(prev_frame)
  cur_img = parse_image(cur_frame)
  return np.concatenate([prev_img, cur_img], axis=0)[None].astype(np.float32)


def input_dtype_from_ort_type(ort_type: str) -> Any:
  import numpy as np

  if ort_type == "tensor(uint8)":
    return np.uint8
  if ort_type in ("tensor(float)", "tensor(float32)"):
    return np.float32
  raise ValueError(f"Unsupported ONNX input type: {ort_type}")


def torch_dtype_from_numpy_dtype(np_dtype: Any) -> Any:
  import numpy as np
  import torch

  if np_dtype == np.uint8:
    return torch.uint8
  if np_dtype == np.float32:
    return torch.float32
  raise ValueError(f"Unsupported numpy dtype: {np_dtype}")


def convert_onnx_to_torch(onnx_path: Path) -> Any:
  import onnx
  from onnx2torch import convert

  onnx_model = onnx.load(str(onnx_path))
  # Some ONNX exporters keep missing optional inputs as trailing empty strings,
  # e.g. Clip(x, min, ""). ONNX Runtime accepts this, but onnx2torch 1.5.x tries
  # to resolve the empty string as a dynamic constant. Drop only trailing blanks.
  for node in onnx_model.graph.node:
    while len(node.input) > 0 and node.input[-1] == "":
      del node.input[-1]
  torch_model = convert(onnx_model)
  torch_model.eval()
  return torch_model


def run_torch_model(torch_model: Any, img_input: Any, input_dtype: Any, device: str) -> Any:
  import torch

  torch_dtype = torch_dtype_from_numpy_dtype(input_dtype)
  img = torch.as_tensor(img_input.astype(input_dtype), dtype=torch_dtype, device=device)

  with torch.no_grad():
    output = torch_model(img, img)

  if isinstance(output, (tuple, list)):
    output = output[0]
  return output.detach().cpu().numpy()


def run_onnx_model(ort_session: Any, img_input: Any, input_dtype: Any) -> Any:
  input_names = [item.name for item in ort_session.get_inputs()]
  if len(input_names) != 2:
    raise ValueError(f"Expected 2 vision inputs, got {len(input_names)}: {input_names}")

  img_input = img_input.astype(input_dtype)
  outputs = ort_session.run(None, {
    input_names[0]: img_input,
    input_names[1]: img_input,
  })
  return outputs[0]


def extract_lead_drel(raw_output: Any) -> Any:
  import numpy as np

  raw_output = np.asarray(raw_output)
  lead_raw = raw_output[:, LEAD_SLICE]
  lead_mu = lead_raw[:, :LEAD_MU_SIZE]
  lead = lead_mu.reshape((-1, 3, 6, 4))
  return lead[:, :, :, 0]


def compare_on_video(args: argparse.Namespace, torch_model: Any) -> None:
  import cv2
  import numpy as np
  import onnxruntime as ort
  import torch

  providers = ["CPUExecutionProvider"]
  ort_session = ort.InferenceSession(str(args.onnx), providers=providers)
  ort_inputs = ort_session.get_inputs()
  input_dtype = input_dtype_from_ort_type(ort_inputs[0].type)

  for ort_input in ort_inputs:
    print(f"ONNX input: {ort_input.name} {ort_input.type} {ort_input.shape}")
  for ort_output in ort_session.get_outputs():
    print(f"ONNX output: {ort_output.name} {ort_output.type} {ort_output.shape}")

  cap = cv2.VideoCapture(str(args.video))
  if not cap.isOpened():
    raise RuntimeError(f"Unable to open video: {args.video}")

  ok, prev_frame = cap.read()
  if not ok:
    raise RuntimeError(f"Unable to read first frame from video: {args.video}")

  raw_error = OutputError()
  lead_error = OutputError()
  printed_first = False
  pair_idx = 0
  read_idx = 1

  torch_model.to(args.device)

  while pair_idx < args.max_pairs:
    ok, cur_frame = cap.read()
    if not ok:
      break

    if read_idx % args.stride != 0:
      prev_frame = cur_frame
      read_idx += 1
      continue

    img_input = make_model_input(prev_frame, cur_frame)

    onnx_output = run_onnx_model(ort_session, img_input, input_dtype)
    torch_output = run_torch_model(torch_model, img_input, input_dtype, args.device)

    if onnx_output.shape != torch_output.shape:
      raise RuntimeError(f"Output shape mismatch: ONNX {onnx_output.shape}, PyTorch {torch_output.shape}")
    if onnx_output.shape[-1] != VISION_OUTPUT_SIZE:
      print(f"Warning: expected output size {VISION_OUTPUT_SIZE}, got {onnx_output.shape[-1]}")

    raw_diff = torch_output - onnx_output
    lead_diff = extract_lead_drel(torch_output) - extract_lead_drel(onnx_output)
    raw_error.update(raw_diff)
    lead_error.update(lead_diff)

    if not printed_first:
      onnx_drel = extract_lead_drel(onnx_output)[0, 0].tolist()
      torch_drel = extract_lead_drel(torch_output)[0, 0].tolist()
      print("\nFirst compared frame pair lead[0].x / dRel:")
      print("  ONNX   :", [round(float(x), 6) for x in onnx_drel])
      print("  PyTorch:", [round(float(x), 6) for x in torch_drel])
      print("  abs err:", [round(float(x), 6) for x in np.abs(np.asarray(torch_drel) - np.asarray(onnx_drel))])
      printed_first = True

    pair_idx += 1
    read_idx += 1
    prev_frame = cur_frame

  cap.release()

  print(f"\nCompared frame pairs: {pair_idx}")
  print("Raw output error:")
  print(f"  MAE     : {raw_error.mae:.8g}")
  print(f"  RMSE    : {raw_error.rmse:.8g}")
  print(f"  Max abs : {raw_error.max_abs:.8g}")
  print("lead.x / dRel error:")
  print(f"  MAE     : {lead_error.mae:.8g}")
  print(f"  RMSE    : {lead_error.rmse:.8g}")
  print(f"  Max abs : {lead_error.max_abs:.8g}")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX_PATH, help="Path to driving_vision.onnx")
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Where to save the PyTorch model")
  parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH, help="Video used for ONNX vs PyTorch comparison")
  parser.add_argument("--max-pairs", type=int, default=20, help="Maximum number of consecutive frame pairs to compare")
  parser.add_argument("--stride", type=int, default=1, help="Compare every Nth frame pair")
  parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda")
  parser.add_argument("--skip-compare", action="store_true", help="Only convert and save the model")
  return parser.parse_args()


def main() -> None:
  require_modules()

  import torch

  args = parse_args()
  args.onnx = args.onnx.resolve()
  args.output = args.output.resolve()
  args.video = args.video.resolve()

  if not args.onnx.exists():
    raise FileNotFoundError(args.onnx)
  if not args.skip_compare and not args.video.exists():
    raise FileNotFoundError(args.video)

  print(f"Converting ONNX model: {args.onnx}")
  torch_model = convert_onnx_to_torch(args.onnx)

  args.output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(torch_model.cpu(), args.output)
  print(f"Saved PyTorch model: {args.output}")

  if not args.skip_compare:
    print(f"\nComparing ONNX vs PyTorch on video: {args.video}")
    compare_on_video(args, torch_model)


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    sys.exit(130)
