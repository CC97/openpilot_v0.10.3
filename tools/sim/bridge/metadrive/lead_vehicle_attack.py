from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_PATH = REPO_ROOT / "tools/sim/bridge/pytorch_model/driving_vision_torch.pt"

MODEL_W = 512
MODEL_H = 256
LEAD_SLICE = slice(917, 1061)
LEAD_PROB_SLICE = slice(1061, 1064)
LEAD_MU_SIZE = 3 * 6 * 4


class AdamOptTorch:
  def __init__(self, size, device, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, dtype=None):
    import torch

    dtype = dtype or torch.float32
    self.exp_avg = torch.zeros(size, dtype=dtype, device=device)
    self.exp_avg_sq = torch.zeros(size, dtype=dtype, device=device)
    self.beta1 = torch.tensor(beta1, dtype=dtype, device=device)
    self.beta2 = torch.tensor(beta2, dtype=dtype, device=device)
    self.eps = eps
    self.lr = lr
    self.step = 0

  def update(self, grad):
    import torch

    self.step += 1
    bias_correction1 = 1 - self.beta1 ** self.step
    bias_correction2 = 1 - self.beta2 ** self.step

    self.exp_avg = self.beta1 * self.exp_avg + (1 - self.beta1) * grad
    self.exp_avg_sq = self.beta2 * self.exp_avg_sq + (1 - self.beta2) * (grad ** 2)

    denom = (torch.sqrt(self.exp_avg_sq) / torch.sqrt(bias_correction2)) + self.eps
    step_size = self.lr / bias_correction1
    return step_size / denom * self.exp_avg


@dataclass
class LeadVehicleAttackConfig:
  enabled: bool = True
  model_path: str = str(DEFAULT_MODEL_PATH)
  device: str = "auto"
  mask_iterations: int = 10
  optimize_every_n_frames: int = 20
  thres: float = 1.0
  lr: float = 1.0
  min_bbox_area: int = 16


class LeadVehiclePatchAttack:
  def __init__(self, config: dict | None = None):
    self.config = LeadVehicleAttackConfig(**(config or {}))
    self.frame_idx = 0
    self.prev_sent_rgb: np.ndarray | None = None
    self.patch = None
    self.patch_bounds: tuple[int, int, int, int] | None = None
    self.model = None
    self.torch = None
    self.device = None

  def apply(self, rgb: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    self.frame_idx += 1
    if not self.config.enabled or bbox is None:
      self.prev_sent_rgb = rgb.copy()
      return rgb

    bbox = self._clip_bbox(bbox, rgb.shape)
    if bbox is None:
      self.prev_sent_rgb = rgb.copy()
      return rgb

    should_optimize = (
      self.patch is None or
      self.patch_bounds is None or
      self.frame_idx % max(1, self.config.optimize_every_n_frames) == 0
    )
    if self.prev_sent_rgb is not None and should_optimize:
      self.patch, self.patch_bounds = self._optimize_patch(self.prev_sent_rgb, rgb, bbox)

    patched_rgb = self._apply_patch_to_rgb(rgb, bbox)
    self.prev_sent_rgb = patched_rgb.copy()
    return patched_rgb

  def _clip_bbox(self, bbox: tuple[int, int, int, int], shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, x1 + 1, w))
    y2 = int(np.clip(y2, y1 + 1, h))
    if (x2 - x1) * (y2 - y1) < self.config.min_bbox_area:
      return None
    return x1, y1, x2, y2

  def _load_model(self):
    if self.model is not None:
      return

    import torch

    self.torch = torch
    if self.config.device == "auto":
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = self.config.device
    model_path = Path(self.config.model_path)
    self.model = torch.load(model_path, map_location=self.device, weights_only=False)
    self.model.eval()
    self.model.to(self.device)
    for param in self.model.parameters():
      param.requires_grad_(False)

  def _bbox_to_model_y_bounds(self, bbox: tuple[int, int, int, int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    h0 = int(y1 * MODEL_H / h)
    h1 = int(np.ceil(y2 * MODEL_H / h))
    w0 = int(x1 * MODEL_W / w)
    w1 = int(np.ceil(x2 * MODEL_W / w))
    h0 = int(np.clip(h0, 0, MODEL_H - 1))
    w0 = int(np.clip(w0, 0, MODEL_W - 1))
    h1 = int(np.clip(max(h0 + 1, h1), h0 + 1, MODEL_H))
    w1 = int(np.clip(max(w0 + 1, w1), w0 + 1, MODEL_W))
    return h0, h1, w0, w1

  def _rgb_to_model_yuv(self, rgb: np.ndarray) -> np.ndarray:
    rgb_small = cv2.resize(rgb, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(rgb_small, cv2.COLOR_RGB2YUV_I420).astype(np.float32)

  def _parse_image_multi(self, frames):
    torch = self.torch
    n = frames.shape[0]
    h = (frames.shape[1] * 2) // 3
    w = frames.shape[2]

    parsed = torch.zeros(size=(n, 6, h // 2, w // 2), dtype=torch.float32, device=frames.device)
    parsed[:, 0] = frames[:, 0:h:2, 0::2]
    parsed[:, 1] = frames[:, 1:h:2, 0::2]
    parsed[:, 2] = frames[:, 0:h:2, 1::2]
    parsed[:, 3] = frames[:, 1:h:2, 1::2]
    parsed[:, 4] = frames[:, h:h + h // 4].reshape((n, h // 2, w // 2))
    parsed[:, 5] = frames[:, h + h // 4:h + h // 2].reshape((n, h // 2, w // 2))
    return parsed

  def _lead_drel_target(self, output):
    torch = self.torch
    if isinstance(output, (tuple, list)):
      output = output[0]

    lead_mu = output[:, LEAD_SLICE][:, :LEAD_MU_SIZE].reshape((-1, 3, 6, 4))
    lead_prob = torch.sigmoid(output[:, LEAD_PROB_SLICE])
    lead_idx = int(torch.argmax(lead_prob[0]).detach().cpu())
    return lead_mu[0, lead_idx, 0, 0]

  def _optimize_patch(self, prev_rgb: np.ndarray, cur_rgb: np.ndarray,
                      bbox: tuple[int, int, int, int]):
    self._load_model()
    torch = self.torch
    h0, h1, w0, w1 = self._bbox_to_model_y_bounds(bbox, cur_rgb.shape)

    base_yuv = np.stack([self._rgb_to_model_yuv(prev_rgb), self._rgb_to_model_yuv(cur_rgb)])
    base_yuv_t = torch.tensor(base_yuv, dtype=torch.float32, device=self.device)

    patch = self.config.thres * torch.rand((h1 - h0, w1 - w0), dtype=torch.float32, device=self.device)
    patch.requires_grad_(True)
    adam = AdamOptTorch(patch.shape, device=self.device, lr=self.config.lr)

    for _ in range(self.config.mask_iterations):
      patch = torch.clip(patch, -self.config.thres, self.config.thres)
      tmp_yuv = base_yuv_t.clone()
      tmp_yuv[:, h0:h1, w0:w1] += patch
      tmp_yuv = torch.clip(tmp_yuv, 0.0, 255.0)

      input_imgs = self._parse_image_multi(tmp_yuv).reshape((1, 12, 128, 256))
      output = self.model(input_imgs, input_imgs)
      target = self._lead_drel_target(output)

      self.model.zero_grad(set_to_none=True)
      patch.retain_grad()
      target.backward()

      update = adam.update(patch.grad)
      patch = patch.clone().detach().requires_grad_(True) + update

    patch = torch.clip(patch, -self.config.thres, self.config.thres)
    return patch.detach().cpu().numpy(), (h0, h1, w0, w1)

  def _apply_patch_to_rgb(self, rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    if self.patch is None:
      return rgb

    x1, y1, x2, y2 = bbox
    patch_rgb = cv2.resize(self.patch, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    out = rgb.copy()
    roi = out[y1:y2, x1:x2].astype(np.float32)
    roi += patch_rgb[:, :, None]
    out[y1:y2, x1:x2] = np.rint(np.clip(roi, 0, 255)).astype(np.uint8)
    return out
