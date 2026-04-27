import math
import time
import numpy as np

from collections import namedtuple
from panda3d.core import Point2, Point3, Vec3
from multiprocessing.connection import Connection

from metadrive.engine.core.engine_core import EngineCore
from metadrive.engine.core.image_buffer import ImageBuffer
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.image_obs import ImageObservation
from metadrive.constants import TerrainProperty
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.policy.idm_policy import IDMPolicy

from openpilot.common.realtime import Ratekeeper

from openpilot.tools.sim.bridge.metadrive.lead_vehicle_attack import LeadVehiclePatchAttack
from openpilot.tools.sim.lib.common import vec3
from openpilot.tools.sim.lib.camerad import W, H

C3_POSITION = Vec3(0.0, 0.8, 1.13)
C3_HPR = Vec3(0, 0,0)


metadrive_simulation_state = namedtuple("metadrive_simulation_state", ["running", "done", "done_info"])
metadrive_vehicle_state = namedtuple("metadrive_vehicle_state", ["velocity", "position", "bearing", "steering_angle"])

def apply_metadrive_patches(arrive_dest_done=True, out_of_road_done=True):
  # The pinned MetaDrive wheel may expect a newer assets version than the usable local assets.
  # Avoid a startup-time network update attempt from killing the simulator when GitHub/proxy returns 502.
  from metadrive.engine.asset_loader import AssetLoader
  AssetLoader.should_update_asset = classmethod(lambda cls: False)

  # By default, metadrive won't try to use cuda images unless it's used as a sensor for vehicles, so patch that in
  def add_image_sensor_patched(self, name: str, cls, args):
    if self.global_config["image_on_cuda"]:# and name == self.global_config["vehicle_config"]["image_source"]:
        sensor = cls(*args, self, cuda=True)
    else:
        sensor = cls(*args, self, cuda=False)
    assert isinstance(sensor, ImageBuffer), "This API is for adding image sensor"
    self.sensors[name] = sensor

  EngineCore.add_image_sensor = add_image_sensor_patched

  # we aren't going to use the built-in observation stack, so disable it to save time
  def observe_patched(self, *args, **kwargs):
    return self.state

  ImageObservation.observe = observe_patched

  # disable destination, we want to loop forever
  def arrive_destination_patch(self, *args, **kwargs):
    return False

  def out_of_road_patch(self, *args, **kwargs):
    return False

  if not arrive_dest_done:
    MetaDriveEnv._is_arrive_destination = arrive_destination_patch

  if not out_of_road_done:
    MetaDriveEnv._is_out_of_road = out_of_road_patch

def metadrive_process(dual_camera: bool, config: dict, camera_array, wide_camera_array, image_lock,
                      controls_recv: Connection, simulation_state_send: Connection, vehicle_state_send: Connection,
                      exit_event, op_engaged, test_duration, test_run):
  arrive_dest_done = config.pop("arrive_dest_done", True)
  out_of_road_done = config.pop("out_of_road_done", True)
  map_region_size = config.pop("map_region_size", None)
  front_vehicle_config = config.pop("front_vehicle", {})
  lead_vehicle_attack_config = config.pop("lead_vehicle_attack", {})
  lead_vehicle_bbox_debug_config = config.pop("lead_vehicle_bbox_debug", {})
  enable_idm_lane_change = config.pop("enable_idm_lane_change", True)
  terrain_recenter_forward_offset = config.pop("terrain_recenter_forward_offset", 0.0)
  terrain_recenter_threshold = config.pop("terrain_recenter_threshold", math.inf)
  if map_region_size is not None:
    TerrainProperty.map_region_size = map_region_size

  apply_metadrive_patches(arrive_dest_done, out_of_road_done)

  road_image = np.frombuffer(camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))
  if dual_camera:
    assert wide_camera_array is not None
    wide_road_image = np.frombuffer(wide_camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

  env = MetaDriveEnv(config)
  lead_attack = LeadVehiclePatchAttack(lead_vehicle_attack_config)
  terrain_center = np.array([0.0, 0.0])

  def get_current_lane_info(vehicle):
    _, lane_info, on_lane = vehicle.navigation._get_current_lane(vehicle)
    lane_idx = lane_info[2] if lane_info is not None else None
    return lane_idx, on_lane

  front_vehicle = None
  front_vehicle_policy = None
  front_vehicle_started = False
  last_ground_truth_log_time = 0.0

  def get_route_lane_ahead(distance):
    ego_lane = env.vehicle.lane
    ego_longitude, _ = ego_lane.local_coordinates(env.vehicle.position)
    target_lane_num = env.vehicle.lane_index[2]
    remaining_distance = distance

    lane_distance_left = ego_lane.length - ego_longitude
    if remaining_distance <= lane_distance_left:
      return ego_lane, ego_longitude + remaining_distance

    remaining_distance -= lane_distance_left
    navigation = env.vehicle.navigation
    checkpoints = navigation.checkpoints
    checkpoint_index = navigation._target_checkpoints_index[1]
    road_network = navigation.map.road_network

    while checkpoint_index < len(checkpoints) - 1:
      route_lanes = road_network.graph[checkpoints[checkpoint_index]][checkpoints[checkpoint_index + 1]]
      route_lane = route_lanes[min(target_lane_num, len(route_lanes) - 1)]
      if remaining_distance <= route_lane.length:
        return route_lane, remaining_distance
      remaining_distance -= route_lane.length
      checkpoint_index += 1

    return None, None

  def spawn_front_vehicle():
    nonlocal front_vehicle, front_vehicle_policy, front_vehicle_started

    if not front_vehicle_config.get("enabled", False):
      return

    spawn_lane, spawn_longitude = get_route_lane_ahead(front_vehicle_config.get("distance", 100.0))
    if spawn_lane is None:
      return

    vehicle_config = dict(env.engine.global_config["traffic_vehicle_config"])
    vehicle_config.update(
      spawn_lane_index=spawn_lane.index,
      spawn_longitude=spawn_longitude,
      spawn_lateral=0.0,
      enable_reverse=False,
      render_vehicle=front_vehicle_config.get("render_vehicle", True),
      #random_color=True,
      #use_special_color=True,
    )

    traffic_manager = env.engine.traffic_manager
    front_vehicle = traffic_manager.spawn_object(
      DefaultVehicle,
      vehicle_config=vehicle_config,
      name="front_vehicle",
    )
    front_vehicle_policy = traffic_manager.add_policy(
      front_vehicle.id, IDMPolicy, front_vehicle, traffic_manager.generate_seed()
    )
    front_vehicle_policy.target_speed = front_vehicle_config.get("target_speed_km_h", front_vehicle_policy.target_speed)
    front_vehicle_started = not front_vehicle_config.get("wait_for_engaged", False)

  def step_front_vehicle():
    nonlocal front_vehicle_started

    if front_vehicle is None or front_vehicle_policy is None:
      return

    if not front_vehicle_started:
      if not op_engaged.is_set():
        front_vehicle.before_step([0.0, -1.0])
        return
      front_vehicle_started = True

    front_vehicle.before_step(front_vehicle_policy.act())

  def recenter_terrain_if_needed(force=False):
    nonlocal terrain_center

    if not env.engine.terrain.render:
      return

    ego_heading = env.vehicle.heading_theta
    forward = np.array([math.cos(ego_heading), math.sin(ego_heading)])
    target_center = np.asarray(env.vehicle.position, dtype=float) + forward * terrain_recenter_forward_offset
    if force or np.linalg.norm(target_center - terrain_center) >= terrain_recenter_threshold:
      env.engine.terrain.before_reset()
      env.engine.terrain.reset(target_center)
      terrain_center = target_center

  def get_front_vehicle_ground_truth():
    if front_vehicle is None:
      return None

    ego_pos = np.asarray(env.vehicle.position, dtype=float)
    ego_vel = np.asarray(env.vehicle.velocity, dtype=float)
    lead_pos = np.asarray(front_vehicle.position, dtype=float)
    lead_vel = np.asarray(front_vehicle.velocity, dtype=float)

    rel_pos = lead_pos - ego_pos
    ego_heading = env.vehicle.heading_theta
    forward = np.array([math.cos(ego_heading), math.sin(ego_heading)])
    left = np.array([-forward[1], forward[0]])

    return {
      "ego_pos": ego_pos,
      "ego_vel": ego_vel,
      "ego_heading": float(ego_heading),
      "lead_pos": lead_pos,
      "lead_vel": lead_vel,
      "lead_heading": float(front_vehicle.heading_theta),
      "lead_distance": float(np.linalg.norm(rel_pos)),
      "lead_rel_longitudinal": float(np.dot(rel_pos, forward)),
      "lead_rel_lateral": float(np.dot(rel_pos, left)),
    }

  def log_front_vehicle_ground_truth():
    ground_truth = get_front_vehicle_ground_truth()
    if ground_truth is None:
      return

    print(
      "MetaDrive GT "
      f"ego_pos=({ground_truth['ego_pos'][0]:.2f},{ground_truth['ego_pos'][1]:.2f}) "
      f"ego_speed={np.linalg.norm(ground_truth['ego_vel']):.2f}m/s "
      f"ego_heading={math.degrees(ground_truth['ego_heading']):.1f}deg "
      f"lead_pos=({ground_truth['lead_pos'][0]:.2f},{ground_truth['lead_pos'][1]:.2f}) "
      f"lead_speed={np.linalg.norm(ground_truth['lead_vel']):.2f}m/s "
      f"lead_heading={math.degrees(ground_truth['lead_heading']):.1f}deg "
      f"rel_long={ground_truth['lead_rel_longitudinal']:.2f}m "
      f"rel_lat={ground_truth['lead_rel_lateral']:.2f}m "
      f"distance={ground_truth['lead_distance']:.2f}m",
      flush=True,
    )

  def reset():
    nonlocal front_vehicle, front_vehicle_policy, front_vehicle_started, terrain_center, last_ground_truth_log_time

    front_vehicle = None
    front_vehicle_policy = None
    front_vehicle_started = False
    terrain_center = np.array([0.0, 0.0])
    last_ground_truth_log_time = 0.0
    env.reset()
    env.engine.global_config["enable_idm_lane_change"] = enable_idm_lane_change
    env.vehicle.config["max_speed_km_h"] = 1000
    recenter_terrain_if_needed(force=True)
    spawn_front_vehicle()
    lane_idx_prev, _ = get_current_lane_info(env.vehicle)

    simulation_state = metadrive_simulation_state(
      running=True,
      done=False,
      done_info=None,
    )
    simulation_state_send.send(simulation_state)

    return lane_idx_prev

  lane_idx_prev = reset()
  start_time = None

  def get_cam_as_rgb(cam):
    cam = env.engine.sensors[cam]
    cam.get_cam().reparentTo(env.vehicle.origin)
    cam.get_cam().setPos(C3_POSITION)
    cam.get_cam().setHpr(C3_HPR)
    img = cam.perceive(to_float=False)
    if not isinstance(img, np.ndarray):
      img = img.get() # convert cupy array to numpy
    return img

  def get_front_vehicle_bbox():
    if front_vehicle is None or "rgb_road" not in env.engine.sensors:
      return None

    cam = env.engine.sensors["rgb_road"]
    cam_np = cam.get_cam()
    cam_np.reparentTo(env.vehicle.origin)
    cam_np.setPos(C3_POSITION)
    cam_np.setHpr(C3_HPR)
    lens = cam.get_lens()

    points = []
    for corner in get_front_vehicle_bbox_corners():
      point_3d = cam_np.getRelativePoint(front_vehicle.origin, Point3(*corner))
      point_2d = Point2()
      if lens.project(point_3d, point_2d):
        points.append((
          (point_2d.x + 1.0) * 0.5 * W,
          (1.0 - point_2d.y) * 0.5 * H,
        ))

    if not points:
      return None

    points = np.asarray(points)
    x1 = int(np.floor(np.clip(points[:, 0].min(), 0, W - 1)))
    y1 = int(np.floor(np.clip(points[:, 1].min(), 0, H - 1)))
    x2 = int(np.ceil(np.clip(points[:, 0].max(), x1 + 1, W)))
    y2 = int(np.ceil(np.clip(points[:, 1].max(), y1 + 1, H)))
    return x1, y1, x2, y2

  def get_front_vehicle_bbox_corners():
    tight_bounds = None
    try:
      tight_bounds = front_vehicle.origin.getTightBounds(front_vehicle.origin)
    except Exception:
      tight_bounds = None

    if tight_bounds is not None:
      min_p, max_p = tight_bounds
      if min_p is not None and max_p is not None:
        return [
          (x, y, z)
          for x in (min_p.x, max_p.x)
          for y in (min_p.y, max_p.y)
          for z in (min_p.z, max_p.z)
        ]

    return [
      (x, y, z)
      for x in (-front_vehicle.WIDTH / 2.0, front_vehicle.WIDTH / 2.0)
      for y in (-front_vehicle.LENGTH / 2.0, front_vehicle.LENGTH / 2.0)
      for z in (0.0, front_vehicle.HEIGHT)
    ]

  def draw_front_vehicle_bbox(rgb, bbox):
    if not lead_vehicle_bbox_debug_config.get("enabled", False) or bbox is None:
      return rgb

    out = rgb.copy()
    height, width = out.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = int(np.clip(x1, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x2 = int(np.clip(x2, x1 + 1, width))
    y2 = int(np.clip(y2, y1 + 1, height))
    thickness = max(1, int(lead_vehicle_bbox_debug_config.get("thickness", 8)))
    color = np.array([0, 255, 0], dtype=np.uint8)

    if lead_vehicle_bbox_debug_config.get("fill", False):
      out[y1:y2, x1:x2] = color
    else:
      out[y1:min(y1 + thickness, y2), x1:x2] = color
      out[max(y2 - thickness, y1):y2, x1:x2] = color
      out[y1:y2, x1:min(x1 + thickness, x2)] = color
      out[y1:y2, max(x2 - thickness, x1):x2] = color
    return out

  def save_bbox_debug_frame(rgb, bbox, frame_idx):
    if not lead_vehicle_bbox_debug_config.get("enabled", False):
      return

    save_every = int(lead_vehicle_bbox_debug_config.get("save_every_n_frames", 0))
    if save_every > 0 and frame_idx % save_every == 0:
      import cv2
      save_path = lead_vehicle_bbox_debug_config.get("save_path", "/tmp/metadrive_lead_bbox_debug.png")
      cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    log_every = int(lead_vehicle_bbox_debug_config.get("log_every_n_frames", 0))
    if log_every > 0 and frame_idx % log_every == 0:
      print(f"MetaDrive lead bbox debug frame={frame_idx} bbox={bbox}", flush=True)

  rk = Ratekeeper(100, None)

  steer_ratio = 8
  vc = [0,0]
  camera_frame_idx = 0

  while not exit_event.is_set():
    vehicle_state = metadrive_vehicle_state(
      velocity=vec3(x=float(env.vehicle.velocity[0]), y=float(env.vehicle.velocity[1]), z=0),
      position=env.vehicle.position,
      bearing=float(math.degrees(env.vehicle.heading_theta)),
      steering_angle=env.vehicle.steering * env.vehicle.MAX_STEERING
    )
    vehicle_state_send.send(vehicle_state)

    if controls_recv.poll(0):
      while controls_recv.poll(0):
        steer_angle, gas, should_reset = controls_recv.recv()

      steer_metadrive = steer_angle * 1 / (env.vehicle.MAX_STEERING * steer_ratio)
      steer_metadrive = np.clip(steer_metadrive, -1, 1)

      vc = [steer_metadrive, gas]

      if should_reset:
        lane_idx_prev = reset()
        start_time = None

    is_engaged = op_engaged.is_set()
    if is_engaged and start_time is None:
      start_time = time.monotonic()

    if rk.frame % 5 == 0:
      step_front_vehicle()
      _, _, terminated, _, _ = env.step(vc)
      if front_vehicle is not None:
        front_vehicle.after_step()
      if time.monotonic() - last_ground_truth_log_time >= 1.0:
        last_ground_truth_log_time = time.monotonic()
        log_front_vehicle_ground_truth()
      recenter_terrain_if_needed()
      timeout = True if start_time is not None and time.monotonic() - start_time >= test_duration else False
      lane_idx_curr, on_lane = get_current_lane_info(env.vehicle)
      out_of_lane = lane_idx_curr != lane_idx_prev or not on_lane
      lane_idx_prev = lane_idx_curr

      if terminated or ((out_of_lane or timeout) and test_run):
        if terminated:
          done_result = env.done_function("default_agent")
        elif out_of_lane:
          done_result = (True, {"out_of_lane" : True})
        elif timeout:
          done_result = (True, {"timeout" : True})

        simulation_state = metadrive_simulation_state(
          running=False,
          done=done_result[0],
          done_info=done_result[1],
        )
        simulation_state_send.send(simulation_state)

      if dual_camera:
        wide_road_image[...] = get_cam_as_rgb("rgb_wide")
      road_rgb = get_cam_as_rgb("rgb_road")
      lead_bbox = get_front_vehicle_bbox()
      if lead_attack.config.enabled:
        road_rgb = lead_attack.apply(road_rgb, lead_bbox)
      road_rgb = draw_front_vehicle_bbox(road_rgb, lead_bbox)
      camera_frame_idx += 1
      save_bbox_debug_frame(road_rgb, lead_bbox, camera_frame_idx)
      road_image[...] = road_rgb
      image_lock.release()

    rk.keep_time()
