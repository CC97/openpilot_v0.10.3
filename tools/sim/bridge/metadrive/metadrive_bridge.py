import math
from multiprocessing import Queue

from metadrive.component.sensors.base_camera import _cuda_enable
from metadrive.component.map.pg_map import MapGenerateMethod

from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.metadrive.metadrive_common import RGBCameraRoad, RGBCameraWide
from openpilot.tools.sim.bridge.metadrive.metadrive_world import MetaDriveWorld
from openpilot.tools.sim.lib.camerad import W, H


MAP_REGION_SIZE = 2048
STRAIGHT_ROAD_LENGTH = 5000
FRONT_VEHICLE_DISTANCE = 100.0
TERRAIN_RECENTER_FORWARD_OFFSET = 500.0
TERRAIN_RECENTER_THRESHOLD = 800.0


def straight_block(length):
  return {
    "id": "S",
    "pre_block_socket_index": 0,
    "length": length
  }

def curve_block(length, angle=45, direction=0):
  return {
    "id": "C",
    "pre_block_socket_index": 0,
    "length": length,
    "radius": length,
    "angle": angle,
    "dir": direction
  }

def create_map(track_size=STRAIGHT_ROAD_LENGTH):
  return dict(
    type=MapGenerateMethod.PG_MAP_FILE,
    lane_num=2,
    lane_width=4.5,
    config=[
      None,
      straight_block(track_size),
    ]
  )


class MetaDriveBridge(SimulatorBridge):
  TICKS_PER_FRAME = 5

  def __init__(self, dual_camera, high_quality, test_duration=math.inf, test_run=False):
    super().__init__(dual_camera, high_quality)

    self.should_render = False
    self.test_run = test_run
    self.test_duration = test_duration if self.test_run else math.inf

  def spawn_world(self, queue: Queue):
    sensors = {
      "rgb_road": (RGBCameraRoad, W, H, ),
    }

    if self.dual_camera:
      sensors["rgb_wide"] = (RGBCameraWide, W, H)

    config = dict(
      use_render=self.should_render,
      vehicle_config=dict(
        enable_reverse=False,
        render_vehicle=True,
        image_source="rgb_road",
      ),
      sensors=sensors,
      image_on_cuda=_cuda_enable,
      image_observation=True,
      interface_panel=[],
      out_of_route_done=False,
      out_of_road_done=False,
      on_continuous_line_done=False,
      crash_vehicle_done=False,
      crash_object_done=False,
      arrive_dest_done=False,
      traffic_density=0.0, # traffic is incredibly expensive
      front_vehicle=dict(
        enabled=True,
        render_vehicle=True,
        distance=FRONT_VEHICLE_DISTANCE,
        wait_for_engaged=True,
        target_speed_km_h=33.0, # This is for the lead vehicle, not the ego. The ego speed is set by selfdrive/car/cruise.py: V_CRUISE_INITIAL = 56.32 (kph)
      ),
      lead_vehicle_attack=dict(
        enabled=True,
        device="auto",
        mask_iterations=5,
        optimize_every_n_frames=20,
        thres=8.0,
        lr=1.0,
      ),
      lead_vehicle_bbox_debug=dict(
        enabled=False,
        fill=True,
        thickness=8,
        save_path="/tmp/metadrive_lead_bbox_debug.png",
        save_every_n_frames=0,
        log_every_n_frames=20,
      ),
      enable_idm_lane_change=False,
      terrain_recenter_forward_offset=TERRAIN_RECENTER_FORWARD_OFFSET,
      terrain_recenter_threshold=TERRAIN_RECENTER_THRESHOLD,
      map_region_size=MAP_REGION_SIZE,
      map_config=create_map(),
      decision_repeat=1,
      physics_world_step_size=self.TICKS_PER_FRAME/100,
      preload_models=False,
      show_logo=False,
      anisotropic_filtering=False
    )

    return MetaDriveWorld(queue, config, self.test_duration, self.test_run, self.dual_camera)
