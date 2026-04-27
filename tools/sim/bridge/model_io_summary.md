# openpilot v0.10.3 Model I/O Summary

This note summarizes the two driving models used by `selfdrive/modeld/modeld.py` in openpilot v0.10.3:

- `selfdrive/modeld/models/driving_vision.onnx`
- `selfdrive/modeld/models/driving_policy.onnx`

The authoritative shape and slice information comes from:

- `selfdrive/modeld/models/driving_vision_metadata.pkl`
- `selfdrive/modeld/models/driving_policy_metadata.pkl`
- `selfdrive/modeld/parse_model_outputs.py`
- `selfdrive/modeld/constants.py`

## Data Flow

```text
img, big_img
  -> driving_vision
  -> vision outputs, including lead, lead_prob, hidden_state
  -> features_buffer
  -> driving_policy
  -> plan, desire_state
```

Important for attacks: in this version, the lead vehicle prediction is still in the first model, `driving_vision`. The second model, `driving_policy`, consumes the vision `hidden_state` history and outputs the driving plan and desire state.

## Shared Constants

| Name | Value | Meaning |
| --- | ---: | --- |
| `N_FRAMES` | 2 | Number of camera frames stacked per model input |
| `FEATURE_LEN` | 512 | Length of one vision hidden-state feature vector |
| `IDX_N` | 33 | Number of path/lane/edge trajectory points |
| `LEAD_TRAJ_LEN` | 6 | Number of lead trajectory time points |
| `LEAD_WIDTH` | 4 | Lead tuple width: `x, y, v, a` |
| `PLAN_WIDTH` | 15 | Plan tuple width |
| `DESIRE_LEN` | 8 | Desire classes |
| `DESIRE_PRED_LEN` | 4 | Desire prediction horizon count |
| `DESIRE_PRED_WIDTH` | 8 | Desire class count |
| `NUM_LANE_LINES` | 4 | Lane line count |
| `NUM_ROAD_EDGES` | 2 | Road edge count |

Lead time indices:

```text
LEAD_T_IDXS = [0, 2, 4, 6, 8, 10]
LEAD_T_OFFSETS = [0, 2, 4]
```

Plan time indices:

```text
T_IDXS = 10.0 * (idx / 32)^2, idx = 0..32
```

Lane and road-edge distance indices:

```text
X_IDXS = 192.0 * (idx / 32)^2, idx = 0..32
```

## Parser Rules

`parse_model_outputs.py` applies these transformations:

| Parser method | Used by | Meaning |
| --- | --- | --- |
| `parse_binary_crossentropy` | `meta`, `lane_lines_prob`, `lead_prob` | Applies sigmoid |
| `parse_categorical_crossentropy` | `desire_pred`, `desire_state` | Reshapes if needed, then applies softmax on the last axis |
| `parse_mdn` | `pose`, `lead`, `plan`, lanes, edges, etc. | Splits raw output into mean and std. The first half is `mu`, the second half is `exp(log_std)` |

For the current v0.10.3 metadata, both `lead` and `plan` are non-MHP direct outputs:

- `lead` raw length is `144 = 2 * (3 * 6 * 4)`.
- `plan` raw length is `990 = 2 * (33 * 15)`.

So the first half is the prediction mean and the second half is the std parameter.

## `driving_vision` Inputs

Raw input shapes from `driving_vision_metadata.pkl`:

| Input | Shape | Meaning |
| --- | --- | --- |
| `img` | `(1, 12, 128, 256)` | Main camera model input |
| `big_img` | `(1, 12, 128, 256)` | Wide/extra camera model input |

The 12 channels are two stacked frames, each frame encoded as 6 YUV-planar channels. In `modeld.py`, these are produced by `DrivingModelFrame.prepare(...)` and passed to the model as `uint8` tensors. In a PyTorch attack script, they are often represented as float tensors to allow gradients.

## `driving_vision` Outputs

Raw output shape:

```text
outputs: (1, 1576)
```

Raw output slices:

| Slice name | Raw slice | Raw length | Parser | Parsed output | Meaning |
| --- | ---: | ---: | --- | --- | --- |
| `meta` | `[0:55]` | 55 | sigmoid | `(1, 55)` | Driver/action/disengagement-related probabilities, such as engaged, gas/brake press, blinkers, and hard-brake predictions |
| `desire_pred` | `[55:87]` | 32 | softmax | `(1, 4, 8)` | Predicted future desire classes over 4 horizons |
| `pose` | `[87:99]` | 12 | MDN | `pose: (1, 6)`, `pose_stds: (1, 6)` | Camera/ego motion pose estimate |
| `wide_from_device_euler` | `[99:105]` | 6 | MDN | `(1, 3)`, stds `(1, 3)` | Wide-camera to device Euler rotation estimate |
| `road_transform` | `[105:117]` | 12 | MDN | `(1, 6)`, stds `(1, 6)` | Road transform / camera-road geometry estimate |
| `lane_lines` | `[117:645]` | 528 | MDN | `(1, 4, 33, 2)`, stds `(1, 4, 33, 2)` | Four lane-line trajectories over `X_IDXS`; each point stores lateral/vertical line coordinates |
| `lane_lines_prob` | `[645:653]` | 8 | sigmoid | `(1, 8)` | Lane-line existence/confidence logits converted to probabilities |
| `road_edges` | `[653:917]` | 264 | MDN | `(1, 2, 33, 2)`, stds `(1, 2, 33, 2)` | Left/right road-edge trajectories over `X_IDXS` |
| `lead` | `[917:1061]` | 144 | MDN | `(1, 3, 6, 4)`, stds `(1, 3, 6, 4)` | Three lead-vehicle hypotheses over 6 time points; each point is `x, y, v, a` |
| `lead_prob` | `[1061:1064]` | 3 | sigmoid | `(1, 3)` | Probability/confidence for each of the 3 lead hypotheses |
| `hidden_state` | `[1064:1576]` | 512 | none | `(1, 512)` | 512-dimensional vision feature vector passed into the policy model through `features_buffer` |
| `pad` | `[0:]` | 1576 | unused by parser | Metadata placeholder; not used by `Parser` | Placeholder metadata entry; not a real parsed model output |

### `lead` Layout

After parsing:

```text
lead shape = (batch, lead_index, time_index, value_index)
```

The value index is:

| Index | Field | Meaning |
| ---: | --- | --- |
| 0 | `x` | Longitudinal relative distance, often treated as `dRel` |
| 1 | `y` | Lateral relative position |
| 2 | `v` | Relative/lead velocity prediction |
| 3 | `a` | Lead acceleration prediction |

Example:

```python
drel_now = lead[0, 0, 0, 0]   # first lead, t = 0s, x/dRel
drel_2s  = lead[0, 0, 1, 0]   # first lead, t = 2s, x/dRel
```

`fill_model_msg.py` publishes this as:

```text
modelV2.leadsV3[i].x = lead[0, i, :, 0]
modelV2.leadsV3[i].y = lead[0, i, :, 1]
modelV2.leadsV3[i].v = lead[0, i, :, 2]
modelV2.leadsV3[i].a = lead[0, i, :, 3]
modelV2.leadsV3[i].prob = lead_prob[0, i]
```

### `meta` Layout

After sigmoid, `meta` is a 55-element probability vector.

| Field | Indices | Times |
| --- | --- | --- |
| `ENGAGED` | `[0]` | current |
| `GAS_DISENGAGE` | `[1, 7, 13, 19, 25]` | 2, 4, 6, 8, 10 s |
| `BRAKE_DISENGAGE` | `[2, 8, 14, 20, 26]` | 2, 4, 6, 8, 10 s |
| `STEER_OVERRIDE` | `[3, 9, 15, 21, 27]` | 2, 4, 6, 8, 10 s |
| `HARD_BRAKE_3` | `[4, 10, 16, 22, 28]` | 2, 4, 6, 8, 10 s |
| `HARD_BRAKE_4` | `[5, 11, 17, 23, 29]` | 2, 4, 6, 8, 10 s |
| `HARD_BRAKE_5` | `[6, 12, 18, 24, 30]` | 2, 4, 6, 8, 10 s |
| `GAS_PRESS` | `[31, 35, 39, 43, 47, 51]` | 0, 2, 4, 6, 8, 10 s |
| `BRAKE_PRESS` | `[32, 36, 40, 44, 48, 52]` | 0, 2, 4, 6, 8, 10 s |
| `LEFT_BLINKER` | `[33, 37, 41, 45, 49, 53]` | 0, 2, 4, 6, 8, 10 s |
| `RIGHT_BLINKER` | `[34, 38, 42, 46, 50, 54]` | 0, 2, 4, 6, 8, 10 s |

## `driving_policy` Inputs

Raw input shapes from `driving_policy_metadata.pkl`:

| Input | Shape | Meaning |
| --- | --- | --- |
| `desire_pulse` | `(1, 25, 8)` | History buffer of desire rising-edge pulses |
| `traffic_convention` | `(1, 2)` | One-hot traffic convention, usually `[1, 0]` for right-hand traffic and `[0, 1]` for left-hand traffic |
| `features_buffer` | `(1, 25, 512)` | History buffer of `driving_vision` `hidden_state` vectors |

`modeld.py` builds these buffers with `InputQueues`. Every model cycle, it enqueues:

```python
features_buffer <- vision_outputs_dict["hidden_state"]
desire_pulse <- new_desire
```

Then it runs `driving_policy`.

## `driving_policy` Outputs

Raw output shape:

```text
outputs: (1, 1000)
```

Raw output slices:

| Slice name | Raw slice | Raw length | Parser | Parsed output | Meaning |
| --- | ---: | ---: | --- | --- | --- |
| `plan` | `[0:990]` | 990 | MDN | `plan: (1, 33, 15)`, `plan_stds: (1, 33, 15)` | Planned ego trajectory over 33 time points, including position, velocity, acceleration, orientation, and orientation rate |
| `desire_state` | `[990:998]` | 8 | softmax | `(1, 8)` | Current policy desire state probabilities, such as lane-change related desire classes |
| `pad` | `[-2:]`, equivalent to `[998:1000]` | 2 | unused by parser | Padding/unused | Padding/unused output values |

### `plan` Layout

After parsing:

```text
plan shape = (batch, time_index, value_index)
```

The 15 plan values are grouped as:

| Value slice | Field | Shape per time step | Meaning |
| --- | --- | ---: | --- |
| `[0:3]` | `POSITION` | 3 | `x, y, z` position |
| `[3:6]` | `VELOCITY` | 3 | `x, y, z` velocity |
| `[6:9]` | `ACCELERATION` | 3 | `x, y, z` acceleration |
| `[9:12]` | `T_FROM_CURRENT_EULER` | 3 | Orientation Euler values |
| `[12:15]` | `ORIENTATION_RATE` | 3 | Orientation-rate values |

Example:

```python
position_x = plan[0, :, 0]
velocity_x = plan[0, :, 3]
acceleration_x = plan[0, :, 6]
```

`fill_model_msg.py` publishes this as `modelV2.position`, `modelV2.velocity`, `modelV2.acceleration`, `modelV2.orientation`, and `modelV2.orientationRate`.

## Attack-Relevant Notes

For increasing lead relative distance in v0.10.3:

```python
vision_out = vision_model(img, big_img)
lead_raw = vision_out[:, 917:1061]
lead_mu = lead_raw[:, :72]
lead = lead_mu.reshape(1, 3, 6, 4)
drel_now = lead[0, 0, 0, 0]
```

Then maximize `drel_now` or another `lead[..., 0]` target and backpropagate to the image tensor.

The second model is needed for gradients only when the attack target is a policy output such as `plan`, desired acceleration, desired curvature, or `desire_state`.
