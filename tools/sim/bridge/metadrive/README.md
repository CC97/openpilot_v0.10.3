# MetaDrive Bridge Notes

This directory contains the openpilot simulator bridge for MetaDrive.

## Map Size

`metadrive_bridge.py` exposes the generated map region size with:

```python
MAP_REGION_SIZE = 2048
```

This value is passed into the MetaDrive subprocess as `map_region_size`. In `metadrive_process.py`, it is applied before `MetaDriveEnv(config)` is created:

```python
map_region_size = config.pop("map_region_size", None)
if map_region_size is not None:
  TerrainProperty.map_region_size = map_region_size
```

MetaDrive's road network can be longer than the rendered terrain window. The bridge keeps `MAP_REGION_SIZE` in a stable range and recenters the rendered terrain ahead of the ego vehicle as it drives, so long straight-road tests do not require a huge terrain texture.

## Road Layout

The road shape is built in `create_map()`:

```python
STRAIGHT_ROAD_LENGTH = 1000

def create_map(track_size=STRAIGHT_ROAD_LENGTH):
```

The current map is one long straight road:

```text
straight
```

To make the road longer, increase `STRAIGHT_ROAD_LENGTH`. To make the road wider, change:

```python
lane_width=4.5
```

## Termination

MetaDrive normally ends an episode when the ego vehicle is no longer on a lane, even if the visible road continues. The bridge disables this for long straight-road experiments:

```python
out_of_road_done=False
```

`metadrive_process.py` applies this before `MetaDriveEnv(config)` runs by patching `_is_out_of_road()` to return `False`.

## Vehicle Rendering

The ego vehicle render switch is in `vehicle_config`:

```python
vehicle_config=dict(
  enable_reverse=False,
  render_vehicle=False,
  image_source="rgb_road",
),
```

Changing this to `True` makes MetaDrive load the ego vehicle body and tire models. The full MetaDrive assets must be installed first, otherwise errors like this can occur:

```text
OSError: Could not load model file(s): .../assets/models/ferra/right_tire_front.gltf
```

See the repository root `METADRIVE_ASSETS.md` for the full assets download steps.

Be careful when enabling ego vehicle rendering. The camera is attached to `env.vehicle.origin` in `metadrive_process.py`, so rendering the ego vehicle may put the car body inside the camera view.

## Traffic / Front Vehicle

Random traffic is controlled by:

```python
traffic_density=0.0
```

Increasing this value asks MetaDrive to create traffic vehicles, but it does not guarantee one vehicle will appear directly in front of the ego vehicle.

The bridge also supports a deterministic front vehicle. `metadrive_bridge.py` passes:

```python
front_vehicle=dict(
  enabled=True,
  render_vehicle=True,
  distance=FRONT_VEHICLE_DISTANCE,
  wait_for_engaged=True,
  target_speed_km_h=15.0,
)
enable_idm_lane_change=False
```

After each `env.reset()`, `metadrive_process.py` reads the ego vehicle's current lane, spawns one traffic vehicle ahead on the same lane, and assigns MetaDrive's `IDMPolicy` to it. IDM lane changes are disabled so the vehicle keeps driving on that lane. The front vehicle uses a low target speed by default so it remains visible longer during startup.
If `wait_for_engaged=True`, the front vehicle holds position until openpilot first becomes engaged, then IDM takes over.

Traffic vehicles also need full MetaDrive assets if `render_vehicle=True` is used for them.

## Current Rendering Defaults

The bridge keeps these defaults for performance and openpilot camera compatibility:

```python
use_render=self.should_render
render_vehicle=False
traffic_density=0.0
preload_models=False
anisotropic_filtering=False
```

These settings avoid unnecessary 3D model loading and keep the simulator lighter.
