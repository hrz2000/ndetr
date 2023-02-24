import os
import json
import math
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from collections import deque
from copy import deepcopy
import sys
from rdp import rdp
import pickle # pickle的加载模型时候也要路径对

import hydra
from mmcv import Config
from mmdet3d.models import build_model
sys.path.insert(0,os.path.abspath('./'))
from projects.mmdet3d_plugin import *
from projects.mmdet3d_plugin.datasets.nuscenes_dataset import show_results
from mmdet3d.core.bbox import LiDARInstance3DBoxes
import cv2
import torch
import numpy as np
import carla
from projects.mmdet3d_plugin.models.detectors.detr3d import get_k

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from carla_agent_files.agent_utils.filter_functions import *
from carla_agent_files.agent_utils import transfuser_utils

from leaderboard.autoagents import autonomous_agent

from training.PlanT.dataset import generate_batch
from training.Perception.config import GlobalConfig

from carla_agent_files.nav_planner import RoutePlanner_new as RoutePlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
import xml.etree.ElementTree as ET

from carla_agent_files.utils import MapImage, encode_npy_to_pil, PIXELS_PER_METER
from carla_agent_files.utils import lts_rendering
import pygame
from mmdet3d.core.bbox import get_box_type
import time

def get_entry_point():
    return 'PlanTPerceptionAgent'

# SAVE_GIF = os.getenv("SAVE_GIF", 'False').lower() in ('true', '1', 't')
box_type_3d, box_mode_3d = get_box_type('LiDAR')
intri, vl2cam, cam2vl_r, cam2vl_t, cam2vl = get_k()
cam2img = intri @ np.linalg.inv(cam2vl)

def interpolate_trajectory(world_map, waypoints_trajectory, hop_resolution=1.0, max_len=100):
    """
    Given some raw keypoints interpolate a full dense trajectory to be used by the user.
    returns the full interpolated route both in GPS coordinates and also in its original form.
    Args:
        - world: an reference to the CARLA world so we can use the planner
        - waypoints_trajectory: the current coarse trajectory
        - hop_resolution: is the resolution, how dense is the provided trajectory going to be made
    """

    dao = GlobalRoutePlannerDAO(world_map, hop_resolution)
    grp = GlobalRoutePlanner(dao)
    grp.setup()
    # Obtain route plan
    route = []
    for i in range(len(waypoints_trajectory) - 1):  # Goes until the one before the last.
        waypoint = waypoints_trajectory[i]
        waypoint_next = waypoints_trajectory[i + 1]
        if waypoint.x != waypoint_next.x or waypoint.y != waypoint_next.y:
            interpolated_trace = grp.trace_route(waypoint, waypoint_next)
            if len(interpolated_trace) > max_len:
                waypoints_trajectory[i + 1] = waypoints_trajectory[i]
            else:
                for wp_tuple in interpolated_trace:
                    route.append((wp_tuple[0].transform, wp_tuple[1]))

    lat_ref, lon_ref = _get_latlon_ref(world_map)

    return location_route_to_gps(route, lat_ref, lon_ref), route


def location_route_to_gps(route, lat_ref, lon_ref):
    """
        Locate each waypoint of the route into gps, (lat long ) representations.
    :param route:
    :param lat_ref:
    :param lon_ref:
    :return:
    """
    gps_route = []

    for transform, connection in route:
        gps_point = _location_to_gps(lat_ref, lon_ref, transform.location)
        gps_route.append((gps_point, connection))

    return gps_route


def _get_latlon_ref(world_map):
    """
    Convert from waypoints world coordinates to CARLA GPS coordinates
    :return: tuple with lat and lon coordinates
    """
    xodr = world_map.to_opendrive()
    tree = ET.ElementTree(ET.fromstring(xodr))

    # default reference
    lat_ref = 42.0
    lon_ref = 2.0

    for opendrive in tree.iter('OpenDRIVE'):
        for header in opendrive.iter('header'):
            for georef in header.iter('geoReference'):
                if georef.text:
                    str_list = georef.text.split(' ')
                    for item in str_list:
                        if '+lat_0' in item:
                            lat_ref = float(item.split('=')[1])
                        if '+lon_0' in item:
                            lon_ref = float(item.split('=')[1])
    return lat_ref, lon_ref


def _location_to_gps(lat_ref, lon_ref, location):
    """
    Convert from world coordinates to GPS coordinates
    :param lat_ref: latitude reference for the current map
    :param lon_ref: longitude reference for the current map
    :param location: location to translate
    :return: dictionary with lat, lon and height
    """

    EARTH_RADIUS_EQUA = 6378137.0  # pylint: disable=invalid-name
    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    my = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    mx += location.x
    my -= location.y

    lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
    lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
    z = location.z

    return {'lat': lat, 'lon': lon, 'z': z}




class PlanTPerceptionAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file, route_index=None, cfg=None, exec_or_inter=None):
        self.cfg = cfg
        self.img_shape = (self.cfg.camera_height,self.cfg.camera_width)
        self.route_index = route_index
        self.pos_dict = {}
        
        self.save_freq = 10
        self.save_path = Path(f"{self.cfg.data_save_path}/route_{self.route_index}")

        Path(self.save_path / 'vis').mkdir(parents=True, exist_ok=True) 
        Path(self.save_path / 'topdown').mkdir(parents=True, exist_ok=True)
        
        # hydra.core.global_hydra.GlobalHydra.instance().clear()
        # initialize(config_path="config", job_name="test_app")
        # cfg = compose(config_name="config")
        # print(OmegaConf.to_yaml(cfg))
        self.step = 0
        self.initialized = False
        # self.cfg = cfg.experiments
        self.cnt = 0
        
        self.stuck_detector = 0
        self.forced_move = 0
        self.use_lidar_safe_check = True
        
        torch.cuda.empty_cache()
        self.track = autonomous_agent.Track.MAP
        
        # first args than super setup is important!
        # args_file = open(os.path.join(path_to_conf_file, f'{self.cfg.model_ckpt_load_path}/log/args.txt'), 'r')
        args_file = open(os.path.join(path_to_conf_file, 'args.txt'), 'r')
        self.args = json.load(args_file)
        args_file.close()
        self.cfg_agent = OmegaConf.create(self.args)
        self.config = GlobalConfig(setting='eval')

        self.steer_damping = self.config.steer_damping
        # self.perception_agent = PerceptionAgent(Path(f'{path_to_conf_file}/{self.cfg.perception_ckpt_load_path}'))
        # self.perception_agent.cfg = self.cfg
        
        # super().setup(path_to_conf_file, route_index, cfg, exec_or_inter)

        # print(f'Saving gif: {SAVE_GIF}')
        
        # Filtering
        self.points = MerweScaledSigmaPoints(n=4,
                                            alpha=.00001,
                                            beta=2,
                                            kappa=0,
                                            subtract=residual_state_x)
        self.ukf = UKF(dim_x=4,
                    dim_z=4,
                    fx=bicycle_model_forward,
                    hx=measurement_function_hx,
                    dt=self.config.carla_frame_rate,
                    points=self.points,
                    x_mean_fn=state_mean,
                    z_mean_fn=measurement_mean,
                    residual_x=residual_state_x,
                    residual_z=residual_measurement_h)

        # State noise, same as measurement because we
        # initialize with the first measurement later
        self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
        # Measurement noise
        self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
        self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise
        # Used to set the filter state equal the first measurement
        self.filter_initialized = False
        # Stores the last filtered positions of the ego vehicle.
        # Used to realign.
        self.state_log = deque(maxlen=4)

        # exec_or_inter is used for the interpretability metric
        # exec is the model that executes the actions in carla
        # inter is the model that obtains attention scores and a ranking of the vehicles importance
        # if exec_or_inter is not None:
        #     if exec_or_inter == 'exec':
        #         LOAD_CKPT_PATH = cfg.exec_model_ckpt_load_path
        #     elif exec_or_inter == 'inter':
        #         LOAD_CKPT_PATH = cfg.inter_model_ckpt_load_path
        # else:
        # LOAD_CKPT_PATH = f'{path_to_conf_file}/{self.cfg.model_ckpt_load_path}/checkpoints/epoch=0{self.cfg.PlanT_epoch}.ckpt'

        # print(f'Loading model from {LOAD_CKPT_PATH}')

        # if Path(LOAD_CKPT_PATH).suffix == '.ckpt':
        #     self.net = LitHFLM.load_from_checkpoint(LOAD_CKPT_PATH)
        # else:
        #     raise Exception(f'Unknown model type: {Path(LOAD_CKPT_PATH).suffix}')
        config = self.cfg.mmdet_cfg
        mmdet_cfg = Config.fromfile(config)

        self.net = build_model(
            mmdet_cfg.model,
            train_cfg=mmdet_cfg.get('train_cfg'),
            test_cfg=mmdet_cfg.get('test_cfg'))
        from mmcv.runner.checkpoint import load_checkpoint
        load_checkpoint(self.net, self.cfg.weight, strict=False)
        self.net = self.net.to('cuda')

        self.net.eval()

    def get_nearby_object(self, vehicle_position, actor_list, radius):
        nearby_objects = []
        for actor in actor_list:
            trigger_box_global_pos = actor.get_transform().transform(actor.trigger_volume.location)
            trigger_box_global_pos = carla.Location(x=trigger_box_global_pos.x, y=trigger_box_global_pos.y, z=trigger_box_global_pos.z)
            if (trigger_box_global_pos.distance(vehicle_position) < radius):
                nearby_objects.append(actor)
        return nearby_objects

    def lateral_shift(self, transform, shift):
        transform.rotation.yaw += 90
        transform.location += shift * transform.get_forward_vector()
        return transform

    def _init(self, input_data):
        # super()._init(hd_map)
        self.world_map = carla.Map("RouteMap", input_data[1]['opendrive'])
        trajectory = [item[0].location for item in self._global_plan_world_coord]
        self.dense_route, _ = interpolate_trajectory(self.world_map, trajectory)
        
        print("Sparse Waypoints:", len(self._global_plan))
        print("Dense Waypoints:", len(self.dense_route))

        self._waypoint_planner = RoutePlanner(3.5, 50)
        self._waypoint_planner.set_route(self.dense_route, True)
        self._waypoint_planner.save()
        ##################box:
        topology = [x[0] for x in self.world_map.get_topology()]
        topology = sorted(topology, key=lambda w: w.transform.location.z)
        self.polygons = []
        for waypoint in topology:
            waypoints = [waypoint]
            nxt = waypoint.next(self.map_precision)[0]
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt = nxt.next(self.map_precision)[0]

            left_marking = [self.lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
            right_marking = [self.lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]
            self.polygons.append(left_marking + [x for x in reversed(right_marking)])
        #################

        self.hd_map = carla.Map('RouteMap', input_data[1]['opendrive'])
        global_plan_world_coord_positions = []
        for point in self._global_plan_world_coord:
            global_plan_world_coord_positions.append(point[0].location)

        new_trajectory = interpolate_trajectory(self.hd_map, global_plan_world_coord_positions)
        self.hd_map_planner = RoutePlanner(self.config.dense_route_planner_min_distance,
                                       self.config.dense_route_planner_max_distance)
        self.hd_map_planner.set_route(new_trajectory[0], True)
        
        
        self._route_planner = RoutePlanner(self.config.route_planner_min_distance,
                                       self.config.route_planner_max_distance)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        # manually need to set global_route:
        # self.perception_agent._global_plan_world_coord = self._global_plan_world_coord
        # self.perception_agent._global_plan = self._global_plan
        self.save_mask = []
        self.save_topdowns = []
        self.timings_run_step = []
        self.timings_forward_model = []

        self.keep_ids = None

        self.initialized = True
        self.control = carla.VehicleControl()
        self.control.steer = 0.0
        self.control.throttle = 0.0
        self.control.brake = 1.0

        # create map for renderer
        map_image = MapImage(self._world, self.world_map, PIXELS_PER_METER)#5
        make_image = lambda x: np.swapaxes(pygame.surfarray.array3d(x), 0, 1).mean(axis=-1)
        road = make_image(map_image.map_surface)
        lane = make_image(map_image.lane_surface)
        
        self.global_map = np.zeros((1, 15,) + road.shape)
        self.global_map[:, 0, ...] = road / 255.
        self.global_map[:, 1, ...] = lane / 255.

        self.global_map = torch.tensor(self.global_map, device='cuda', dtype=torch.float32)
        world_offset = torch.tensor(map_image._world_offset, device='cuda', dtype=torch.float32)
        self.map_dims = self.global_map.shape[2:4]

        self.renderer = lts_rendering.Renderer(world_offset, self.map_dims, data_generation=True)


    def sensors(self):
        # result = super().sensors()
        result = [
                    {
                        'type': 'sensor.camera.rgb',
                        'x': self.cfg.camera_pos[0],
                        'y': self.cfg.camera_pos[1],
                        'z': self.cfg.camera_pos[2],
                        'roll': self.cfg.camera_rot_0[0],
                        'pitch': self.cfg.camera_rot_0[1],
                        'yaw': self.cfg.camera_rot_0[2],
                        'width': self.cfg.camera_width,
                        'height': self.cfg.camera_height,
                        'fov': self.cfg.camera_fov_data_collection,
                        'id': 'rgb_front'
                    },
                    # {
                    #     'type': 'sensor.lidar.ray_cast',
                    #     'x': self.cfg.lidar_pos[0], 
                    #     'y': self.cfg.lidar_pos[1], 
                    #     'z': self.cfg.lidar_pos[2],
                    #     'roll': self.cfg.lidar_rot[0], 
                    #     'pitch': self.cfg.lidar_rot[1], 
                    #     'yaw': self.cfg.lidar_rot[2],
                    #     'id': 'lidar'
                    #     }, TODO
                    {
                        'type': 'sensor.lidar.ray_cast',
                        'x': self.cfg.lidar_pos[0],
                        'y': self.cfg.lidar_pos[1],
                        'z': self.cfg.lidar_pos[2],
                        'roll': self.cfg.lidar_rot[0],
                        'pitch': self.cfg.lidar_rot[1],
                        'yaw': self.cfg.lidar_rot[2],
                        'rotation_frequency': self.cfg.lidar_rotation_frequency,
                        'points_per_second': self.cfg.lidar_points_per_second,
                        'id': 'lidar'
                    },
                    {
                        'type': 'sensor.other.imu',
                        'x': 0.0,
                        'y': 0.0,
                        'z': 0.0,
                        'roll': 0.0,
                        'pitch': 0.0,
                        'yaw': 0.0,
                        'sensor_tick': self.config.carla_frame_rate,
                        'id': 'imu'
                    }, {
                        'type': 'sensor.other.gnss',
                        'x': 0.0,
                        'y': 0.0,
                        'z': 0.0,
                        'roll': 0.0,
                        'pitch': 0.0,
                        'yaw': 0.0,
                        'sensor_tick': 0.01,
                        'id': 'gps'
                    }, {
                        'type': 'sensor.speedometer',
                        'reading_frequency': self.config.carla_fps,
                        'id': 'speed'
                    }, {
                        'type': 'sensor.opendrive_map',
                        'reading_frequency': 1e-6,
                        'id': 'hd_map'
                    }
        ]
        return result
    
    def _get_position(self, gps):
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps


    def tick(self, input_data, future_wp=None):

        rgb = []
        rgb.append(cv2.cvtColor(input_data['rgb_front'][1][:, :, :3], cv2.COLOR_BGR2RGB))
        rgb = np.concatenate(rgb, axis=1)
        
        lidar = input_data['lidar'][1][:, :3]
        
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = transfuser_utils.preprocess_compass(input_data['imu'][1][-1])
            
        pos = self._route_planner.convert_gps_to_carla(input_data['gps'][1][:2])
        
        pos_old = self._get_position(gps)
        
        compass_old = input_data['imu'][1][-1]
        if (np.isnan(compass_old) == True): # CARLA 0.9.10 occasionally sends NaN values in the compass
            compass_old = 0.0
            
        result = {
                'rgb': rgb,
                'lidar': lidar,
                'gps': pos,
                'gps_old': pos_old,
                'speed': speed,
                'compass': compass,
                'compass_old': compass_old,
                }
        
        
        if not self.filter_initialized:
            self.ukf.x = np.array([pos[0], pos[1], compass, speed])
            self.filter_initialized = True
            
        self.ukf.predict(steer=self.control.steer,
                        throttle=self.control.throttle,
                        brake=self.control.brake)
        self.ukf.update(np.array([pos[0], pos[1], compass, speed]))
        filtered_state = self.ukf.x
        self.state_log.append(filtered_state)
        
        result['gps'] = filtered_state[0:2]

        waypoint_route = self._route_planner.run_step(filtered_state[0:2])
        
        if len(waypoint_route) > 2:
            target_point, _ = waypoint_route[1]
            next_target_point, _ = waypoint_route[2]
        elif len(waypoint_route) > 1:
            target_point, _ = waypoint_route[1]
            next_target_point, _ = waypoint_route[1]
        else:
            target_point, _ = waypoint_route[0]
            next_target_point, _ = waypoint_route[0]
            
        next_wp, next_cmd = waypoint_route[1] if len(
            waypoint_route) > 1 else waypoint_route[0]
        result['next_command'] = next_cmd.value

        theta = compass + np.pi / 2
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])

        local_command_point = np.array(
            [next_wp[0] - filtered_state[0], next_wp[1] - filtered_state[1]])
        local_command_point = rotation_matrix.T.dot(local_command_point)
        # result['target_point_old'] = local_command_point
        
        ego_target_point_raw = transfuser_utils.inverse_conversion_2d(target_point, result['gps'], result['compass'])
        result['target_point_single'] = tuple(ego_target_point_raw)
        
        ego_target_point = torch.from_numpy(ego_target_point_raw[np.newaxis]).to('cuda', dtype=torch.float32)
        if self.config.use_second_tp:
            ego_next_target_point = transfuser_utils.inverse_conversion_2d(next_target_point, result['gps'],
                                                                            result['compass'])
            ego_next_target_point = torch.from_numpy(ego_next_target_point[np.newaxis]).to('cuda', dtype=torch.float32)
            ego_target_point_double = torch.cat((ego_target_point, ego_next_target_point), dim=1)

        result['target_point'] = ego_target_point_double
        
        waypoints_hd = self.hd_map_planner.run_step(filtered_state[0:2])
        self.waypoint_route = np.array([[node[0][0],node[0][1]] for node in waypoints_hd])
        max_len = 50
        if len(self.waypoint_route) < max_len:
            max_len = len(self.waypoint_route)
        
        route_wps = []
        for route in self.waypoint_route[:max_len]:
            route = transfuser_utils.inverse_conversion_2d(route, result['gps'], result['compass'])
            route_wps.append(route * np.array([1,-1]))
            
        result['route_wp'] = route_wps

        return result


    @torch.no_grad()
    def run_step(self, input_data, timestamp, keep_ids=None):
        
        self.keep_ids = keep_ids

        self.step += 1
        if not self.initialized:
            # Privileged: autopliot的init中
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            self._vehicle = CarlaDataProvider.get_hero_actor()
            self._world = self._vehicle.get_world()

            # autopilot
            self.detection_radius = 50.0                    # Distance of obstacles (in meters) in which we will check for collisions
            self.light_radius = 15.0                        # Distance of traffic lights considered relevant (in meters)

            # data_agent_box
            self.map_precision = 10.0 # meters per point
            self.rdp_epsilon = 0.5 # epsilon for route shortening

            self.max_actor_distance = self.detection_radius # copy from expert
            self.max_light_distance = self.light_radius # copy from expert
            self.max_route_distance = 30.0
            self.max_map_element_distance = 30.0

            if 'hd_map' in input_data.keys():
                self._init(input_data['hd_map'])
            else:
                return carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            
            # self._init()
            self.control = carla.VehicleControl()
            self.control.steer = 0.0
            self.control.throttle = 0.0
            self.control.brake = 1.0
            # if self.exec_or_inter == 'inter':
            #     return [], None
            _ = self.tick(input_data)
            return self.control
            
        self.traffic_light_hazard_pred = None
        self.traffic_light_hazard = False

        self.control = self._get_control_ndetr(input_data)

        if self.cfg.traffic_light_brake:
            if self.traffic_light_hazard_pred:
                self.control.brake = 1.0
                self.control.throttle = 0.0
                self.control.steer = 0.0
        
        return self.control
    
    def _get_control_ndetr(self, input_data):
        tick_data = self.tick(input_data)
        label_raw = get_bev_boxes(self, input_data=input_data, pos=tick_data['gps'])
        label_raw_pred = None
        gt_velocity = torch.FloatTensor([tick_data['speed']]).unsqueeze(0)
        inp, gt_result = get_batch_inp_ndetr(self, label_raw_pred, label_raw, tick_data)
        with torch.no_grad():
            results = self.net.forward_test(**inp)
            if self.step % self.save_freq == 0: # 10
                vehicle_position = self._vehicle.get_location()
                pos = (int(vehicle_position.x), int(vehicle_position.y))
                if pos not in self.pos_dict:
                    self.pos_dict[pos] = 0
                self.pos_dict[pos] += 1
                # if self.pos_dict[pos]>20:
                #     # self.destroy()
                #     raise Exception("die")
                time_and_vis(self,
                    gt_result=gt_result, 
                    pred_result=results[0]['pts_bbox'],
                    plan=inp['img_metas'][0]['plan'],
                    inp_img=inp['img'][0][0].detach().cpu().numpy().transpose(1,2,0),
                    save_vis=True)
        pts_bbox = results[0]['pts_bbox']
        pred_wp = pts_bbox['attrs_3d'] # torch.Size([4, 2])

        is_stuck = False
        if self.cfg.unblock:
            # unblock
            # divide by 2 because we process every second frame
            # 1100 = 55 seconds * 20 Frames per second, we move for 1.5 second = 30 frames to unblock
            if(self.stuck_detector > self.config.stuck_threshold and self.forced_move < self.config.creep_duration):
                print("Detected agent being stuck. Move for frame: ", self.forced_move)
                is_stuck = True
                self.forced_move += 1


        steer, throttle, brake = self.net.control_pid(pred_wp, gt_velocity, is_stuck)
        # steer, throttle, brake = self.net.control_pid(pred_wp[:1], gt_velocity, is_stuck)

        if brake < 0.05: brake = 0.0
        if throttle > brake: brake = 0.0

        if brake:
            steer *= self.steer_damping
            
        if self.cfg.unblock:
            if(gt_velocity < 0.1): # 0.1 is just an arbitrary low number to threshhold when the car is stopped
                self.stuck_detector += 1
            elif(gt_velocity > 0.1 and is_stuck == False):
                self.stuck_detector = 0
                self.forced_move    = 0
            if is_stuck:
                steer *= self.steer_damping
                
            # safety check
            safety_box = deepcopy(tick_data['lidar'])
            safety_box[:, 1] *= -1  # invert

            # z-axis
            safety_box      = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
            safety_box      = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]

            # y-axis
            safety_box      = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
            safety_box      = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]

            # x-axis
            safety_box      = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
            safety_box      = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        
        if self.cfg.unblock:
            if self.use_lidar_safe_check:
                emergency_stop = (len(safety_box) > 0) #Checks if the List is empty
                if ((emergency_stop == True) and (is_stuck == True)):  # We only use the saftey box when unblocking
                    print("Detected object directly in front of the vehicle. Stopping. Step:", self.step)
                    control.steer = float(steer)
                    control.throttle = float(0.0)
                    control.brake = float(True)
                    # Will overwrite the stuck detector. If we are stuck in traffic we do want to wait it out.
        
        if self.step < 5:
            control.brake = float(1.0)
            control.throttle = float(0.0)

        return control
    
    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def get_relative_transform(self, ego_matrix, vehicle_matrix):
        """
        return the relative transform from ego_pose to vehicle pose
        """
        relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
        rot = ego_matrix[:3, :3].T
        relative_pos = rot @ relative_pos
        
        # transform to right handed system
        relative_pos[1] = - relative_pos[1]

        # transform relative pos to virtual lidar system
        rot = np.eye(3)
        trans = - np.array([1.3, 0.0, 2.5])
        relative_pos = rot @ relative_pos + trans

        return relative_pos
    
    def get_attn_norm_vehicles(self, attn_map):
        if self.cfg.experiments['attention_score'] == 'AllLayer':
            # attention score for CLS token, sum of all heads
            attn_vector = [np.sum(attn_map[i][0,:,0,1:].numpy(), axis=0) for i in range(len(attn_map))]
        else:
            raise NotImplementedError
            
        attn_vector = np.array(attn_vector)
        offset = 0
        # if no vehicle is in the detection range we add a dummy vehicle
        if len(self.data_car) == 0:
            attn_vector = np.asarray([[0.0]])
            offset = 1

        # sum over layers
        attn_vector = np.sum(attn_vector, axis=0)
        
        # remove route elements
        attn_vector = attn_vector[:len(self.data_car)+offset]+0.00001

        # get max attention score for normalization
        # normalization is only for visualization purposes
        max_attn = np.max(attn_vector)
        attn_vector = attn_vector / max_attn
        attn_vector = np.clip(attn_vector, None, 1)
        
        return attn_vector


    def destroy(self):
        # super().destroy()
        torch.cuda.empty_cache()
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        del self.net
        del self.renderer
        # self.perception_agent.destroy()

def render_BEV(self):
    semantic_grid = self.global_map#torch.Size([1, 15, 1622, 1622])
    
    vehicle_position = self._vehicle.get_location()#<carla.libcarla.Location
    ego_pos_list =  [self._vehicle.get_transform().location.x, self._vehicle.get_transform().location.y]
    #[-41.4886360168457, 112.94529724121094]和get_location再xy一样
    ego_yaw_list =  [self._vehicle.get_transform().rotation.yaw/180*np.pi]#-1.57361008964462

    # fetch local birdview per agent
    ego_pos =  torch.tensor([
        self._vehicle.get_transform().location.x, self._vehicle.get_transform().location.y], device='cuda', dtype=torch.float32)
    ego_yaw =  torch.tensor([self._vehicle.get_transform().rotation.yaw/180*np.pi], device='cuda', dtype=torch.float32)
    birdview = self.renderer.get_local_birdview(#<utils.lts_rendering.Renderer
        semantic_grid,
        ego_pos,
        ego_yaw
    )#torch.Size([1, 15, 500, 500])

    self._actors = self._world.get_actors()#<carla.libcarla.ActorList
    vehicles = self._actors.filter('*vehicle*')#<carla.libcarla.ActorList
    for vehicle in vehicles:
        if (vehicle.get_location().distance(self._vehicle.get_location()) < self.detection_radius):#30.0
            if (vehicle.id != self._vehicle.id) or (vehicle.id == self._vehicle.id):
                pos =  torch.tensor([vehicle.get_transform().location.x, vehicle.get_transform().location.y], device='cuda', dtype=torch.float32)
                yaw =  torch.tensor([vehicle.get_transform().rotation.yaw/180*np.pi], device='cuda', dtype=torch.float32)#世界坐标系下
                veh_x_extent = int(max(vehicle.bounding_box.extent.x*2, 1) * PIXELS_PER_METER)
                veh_y_extent = int(max(vehicle.bounding_box.extent.y*2, 1) * PIXELS_PER_METER)

                self.vehicle_template = torch.ones(1, 1, veh_x_extent, veh_y_extent, device='cuda')#torch.Size([1, 1, 24, 10])
                self.renderer.render_agent_bv(
                    birdview,
                    ego_pos,
                    ego_yaw,
                    self.vehicle_template,
                    pos,
                    yaw,
                    channel=5
                )

    ego_pos_batched = []
    ego_yaw_batched = []
    pos_batched = []
    yaw_batched = []
    template_batched = []
    channel_batched = []

    # -----------------------------------------------------------
    # Pedestrian rendering
    # -----------------------------------------------------------
    walkers = self._actors.filter('*walker*')
    for walker in walkers:
        ego_pos_batched.append(ego_pos_list)
        ego_yaw_batched.append(ego_yaw_list)
        pos_batched.append([walker.get_transform().location.x, walker.get_transform().location.y])
        yaw_batched.append([walker.get_transform().rotation.yaw/180*np.pi])
        channel_batched.append(6)
        template_batched.append(np.ones([20, 7]))

    if len(ego_pos_batched)>0:
        ego_pos_batched_torch = torch.tensor(ego_pos_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
        ego_yaw_batched_torch = torch.tensor(ego_yaw_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
        pos_batched_torch = torch.tensor(pos_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
        yaw_batched_torch = torch.tensor(yaw_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
        template_batched_torch = torch.tensor(template_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
        channel_batched_torch = torch.tensor(channel_batched, device='cuda', dtype=torch.float32)

        self.renderer.render_agent_bv_batched(
            birdview,
            ego_pos_batched_torch,
            ego_yaw_batched_torch,
            template_batched_torch,
            pos_batched_torch,
            yaw_batched_torch,
            channel=channel_batched_torch,
        )

    ego_pos_batched = []
    ego_yaw_batched = []
    pos_batched = []
    yaw_batched = []
    template_batched = []
    channel_batched = []

    # -----------------------------------------------------------
    # Traffic light rendering
    # -----------------------------------------------------------
    traffic_lights = self._actors.filter('*traffic_light*')
    for traffic_light in traffic_lights:
        trigger_box_global_pos = traffic_light.get_transform().transform(traffic_light.trigger_volume.location)
        trigger_box_global_pos = carla.Location(x=trigger_box_global_pos.x, y=trigger_box_global_pos.y, z=trigger_box_global_pos.z)
        if (trigger_box_global_pos.distance(vehicle_position) > self.light_radius):
            continue
        ego_pos_batched.append(ego_pos_list)
        ego_yaw_batched.append(ego_yaw_list)
        pos_batched.append([traffic_light.get_transform().location.x, traffic_light.get_transform().location.y])
        yaw_batched.append([traffic_light.get_transform().rotation.yaw/180*np.pi])
        template_batched.append(np.ones([4, 4]))
        if str(traffic_light.state) == 'Green':
            channel_batched.append(4)
        elif str(traffic_light.state) == 'Yellow':
            channel_batched.append(3)
        elif str(traffic_light.state) == 'Red':
            channel_batched.append(2)

    if len(ego_pos_batched)>0:
        ego_pos_batched_torch = torch.tensor(ego_pos_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
        ego_yaw_batched_torch = torch.tensor(ego_yaw_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
        pos_batched_torch = torch.tensor(pos_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
        yaw_batched_torch = torch.tensor(yaw_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
        template_batched_torch = torch.tensor(template_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
        channel_batched_torch = torch.tensor(channel_batched, device='cuda', dtype=torch.int)

        self.renderer.render_agent_bv_batched(
            birdview,
            ego_pos_batched_torch,
            ego_yaw_batched_torch,
            template_batched_torch,
            pos_batched_torch,
            yaw_batched_torch,
            channel=channel_batched_torch,
        )

    return birdview

def get_bev_boxes(self, input_data=None, lidar=None, pos=None):
    # -----------------------------------------------------------
    # Ego vehicle
    # -----------------------------------------------------------

    # add vehicle velocity and brake flag
    ego_location = self._vehicle.get_location()
    ego_transform = self._vehicle.get_transform()
    ego_control   = self._vehicle.get_control()
    ego_velocity  = self._vehicle.get_velocity()
    ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity) # In m/s
    ego_brake = ego_control.brake
    ego_rotation = ego_transform.rotation
    ego_matrix = np.array(ego_transform.get_matrix())
    ego_extent = self._vehicle.bounding_box.extent
    ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z]) * 2.
    ego_yaw =  ego_rotation.yaw/180*np.pi
    relative_yaw = 0
    relative_pos = self.get_relative_transform(ego_matrix, ego_matrix)

    results = []

    # add ego-vehicle to results list
    # the format is category, extent*3, position*3, yaw, points_in_bbox, distance, id
    # the position is in lidar coordinates
    result = {"class": "Car",
                "extent": [ego_dx[2], ego_dx[0], ego_dx[1] ], #TODO:
                "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                "yaw": relative_yaw,
                "num_points": -1, 
                "distance": -1, 
                "speed": ego_speed, 
                "brake": ego_brake,
                "id": int(self._vehicle.id),
            }
    results.append(result)
    
    # -----------------------------------------------------------
    # Other vehicles
    # -----------------------------------------------------------

    self._actors = self._world.get_actors()
    vehicles = self._actors.filter('*vehicle*')
    walkers = self._actors.filter('*walker*')
    tlights = self._actors.filter('*traffic_light*')
    for actorlist, name in [(vehicles, 'Car'), (walkers, 'Pedestrian')]:
        for vehicle in actorlist:
            if (vehicle.get_location().distance(ego_location) < self.max_actor_distance):
                if (vehicle.id != self._vehicle.id):
                    vehicle_rotation = vehicle.get_transform().rotation
                    vehicle_matrix = np.array(vehicle.get_transform().get_matrix())

                    vehicle_extent = vehicle.bounding_box.extent
                    dx = np.array([vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]) * 2.
                    yaw =  vehicle_rotation.yaw/180*np.pi

                    relative_yaw = normalize_angle(yaw - ego_yaw)
                    relative_pos = self.get_relative_transform(ego_matrix, vehicle_matrix)

                    vehicle_transform = vehicle.get_transform()
                    vehicle_control   = vehicle.get_control()
                    vehicle_velocity  = vehicle.get_velocity()
                    vehicle_speed = self._get_forward_speed(transform=vehicle_transform, velocity=vehicle_velocity) # In m/s
                    if name == 'Car':
                        vehicle_brake = vehicle_control.brake
                    else:
                        vehicle_brake = False

                    # filter bbox that didn't contains points of contains less points
                    if not lidar is None:
                        num_in_bbox_points = self.get_points_in_bbox(ego_matrix, vehicle_matrix, dx, lidar)
                        #print("num points in bbox", num_in_bbox_points)
                    else:
                        num_in_bbox_points = -1

                    distance = np.linalg.norm(relative_pos)

                    result = {
                        "class": name,
                        "extent": [dx[2], dx[0], dx[1]], #TODO
                        "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                        "yaw": relative_yaw,
                        "num_points": int(num_in_bbox_points), 
                        "distance": distance, 
                        "speed": vehicle_speed, 
                        "brake": vehicle_brake,
                        "id": int(vehicle.id),
                    }
                    if name=='Pedestrian':
                        result['position'][2] -= result['extent'][0]/2
                    results.append(result)

    # -----------------------------------------------------------
    # Route rdp
    # -----------------------------------------------------------
    if input_data is not None:
        # pos = self._get_position(input_data['gps'][1][:2])
        # self.gps_buffer.append(pos)
        # pos = np.average(self.gps_buffer, axis=0)  # Denoised position
        self._waypoint_planner.load()
        waypoint_route = self._waypoint_planner.run_step(pos)
        self.waypoint_route = np.array([[node[0][0],node[0][1]] for node in waypoint_route])
        self._waypoint_planner.save()
    
    
    max_len = 50
    if len(self.waypoint_route) < max_len:
        max_len = len(self.waypoint_route)
    shortened_route = rdp(self.waypoint_route[:max_len], epsilon=self.rdp_epsilon)
    
    # convert points to vectors
    vectors = shortened_route[1:] - shortened_route[:-1]
    midpoints = shortened_route[:-1] + vectors/2.
    norms = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:,1], vectors[:,0])

    for i, midpoint in enumerate(midpoints):
        # find distance to center of waypoint
        center_bounding_box = carla.Location(midpoint[0], midpoint[1], 0.0)
        transform = carla.Transform(center_bounding_box)
        route_matrix = np.array(transform.get_matrix())
        relative_pos = self.get_relative_transform(ego_matrix, route_matrix)
        distance = np.linalg.norm(relative_pos)
        
        # find distance to beginning of bounding box
        starting_bounding_box = carla.Location(shortened_route[i][0], shortened_route[i][1], 0.0)
        st_transform = carla.Transform(starting_bounding_box)
        st_route_matrix = np.array(st_transform.get_matrix())
        st_relative_pos = self.get_relative_transform(ego_matrix, st_route_matrix)
        st_distance = np.linalg.norm(st_relative_pos)


        # only store route boxes that are near the ego vehicle
        if i > 0 and st_distance > self.max_route_distance:
            continue

        length_bounding_box = carla.Vector3D(norms[i]/2., ego_extent.y, ego_extent.z)
        bounding_box = carla.BoundingBox(transform.location, length_bounding_box)
        bounding_box.rotation = carla.Rotation(pitch = 0.0,
                                            yaw   = angles[i] * 180 / np.pi,
                                            roll  = 0.0)

        route_extent = bounding_box.extent
        dx = np.array([route_extent.x, route_extent.y, route_extent.z]) * 2.
        relative_yaw = normalize_angle(angles[i] - ego_yaw)

        # visualize subsampled route
        # self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1,
        #                             color=carla.Color(0, 255, 255, 255), life_time=(10.0/self.frame_rate_sim))

        result = {
            "class": "Route",
            "extent": [dx[2], dx[0], dx[1]], #TODO
            "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
            "yaw": relative_yaw,
            "centre_distance": distance,
            "starting_distance": st_distance,
            "id": i,
        }
        results.append(result)


    if int(os.environ.get('DATAGEN')):
        # -----------------------------------------------------------
        # Traffic lights
        # -----------------------------------------------------------

        _traffic_lights = self.get_nearby_object(ego_location, tlights, self.max_light_distance)
    
        for light in _traffic_lights:
            if   (light.state == carla.libcarla.TrafficLightState.Red):
                state = 0
            elif (light.state == carla.libcarla.TrafficLightState.Yellow):
                state = 1 
            elif (light.state == carla.libcarla.TrafficLightState.Green):
                state = 2
            else: # unknown
                state = -1
    
            center_bounding_box = light.get_transform().transform(light.trigger_volume.location)
            center_bounding_box = carla.Location(center_bounding_box.x, center_bounding_box.y, center_bounding_box.z)
            length_bounding_box = carla.Vector3D(light.trigger_volume.extent.x, light.trigger_volume.extent.y, light.trigger_volume.extent.z)
            transform = carla.Transform(center_bounding_box) # can only create a bounding box from a transform.location, not from a location
            bounding_box = carla.BoundingBox(transform.location, length_bounding_box)

            gloabl_rot = light.get_transform().rotation
            bounding_box.rotation = carla.Rotation(pitch = light.trigger_volume.rotation.pitch + gloabl_rot.pitch,
                                                yaw   = light.trigger_volume.rotation.yaw   + gloabl_rot.yaw,
                                                roll  = light.trigger_volume.rotation.roll  + gloabl_rot.roll)
            
            light_rotation = transform.rotation
            light_matrix = np.array(transform.get_matrix())

            light_extent = bounding_box.extent
            dx = np.array([light_extent.x, light_extent.y, light_extent.z]) * 2.
            yaw =  light_rotation.yaw/180*np.pi

            relative_yaw = normalize_angle(yaw - ego_yaw)
            relative_pos = self.get_relative_transform(ego_matrix, light_matrix)

            distance = np.linalg.norm(relative_pos)

            result = {
                "class": "Light",
                "extent": [dx[2], dx[0], dx[1]], #TODO
                "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                "yaw": relative_yaw,
                "distance": distance, 
                "state": state, 
                "id": int(light.id),
            }
            results.append(result)

        # -----------------------------------------------------------
        # Map elements
        # -----------------------------------------------------------
        for lane_id, poly in enumerate(self.polygons):
            for point_id, point in enumerate(poly):
                if (point.location.distance(ego_location) < self.max_map_element_distance):
                    point_matrix = np.array(point.get_matrix())

                    yaw =  point.rotation.yaw/180*np.pi

                    relative_yaw = yaw - ego_yaw
                    relative_pos = self.get_relative_transform(ego_matrix, point_matrix)
                    distance = np.linalg.norm(relative_pos)

                    result = {
                        "class": "Lane",
                        "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                        "yaw": relative_yaw,
                        "distance": distance,
                        "point_id": int(point_id),
                        "lane_id": int(lane_id),
                    }
                    results.append(result)
                
    return results

from tools.preprocess.carla_dataset import extract_data, extract_data2
def get_input_batch_plant(label_raw, input_data):
    sample = {'input': [], 'output': [], 'brake': [], 'waypoints': [], 'target_point': [], 'light': []}

    # label_raw 包含ego
    data_car, data_route, obj_idxs = extract_data(label_raw, max_NextRouteBBs=2)

    # if self.keep_ids is not None:
    #     data_car = [x for i, x in enumerate(data_car) if i in self.keep_ids]
    #     assert len(data_car) <= len(self.keep_ids), f'{len(data_car)} <= {len(self.keep_ids)}'

    features = data_car + data_route

    inp = np.array(features)
    instance_t, gt_labels_3d, data_route = extract_data2(inp) # 本来就是lidar系下

    gt_result = dict(
        boxes_3d = LiDARInstance3DBoxes(instance_t, box_dim=instance_t.shape[-1]),
        labels_3d = torch.tensor(gt_labels_3d.astype(int)).to('cuda'),
        scores_3d = np.ones(gt_labels_3d.shape[0])
    )

    features = inp[:,:7]

    sample['input'] = features
    sample['output'] = features

    # sample['light'] = self.traffic_light_hazard_pred
    sample['light'] = 0 # light总是0, 不输入gt light信息 TODO
    # import pdb;pdb.set_trace()
    if 'target_point_single' in input_data: # 感知那一系列本身就有
        local_command_point = np.array([input_data['target_point_single'][0], input_data['target_point_single'][1]])
    elif 'my_labels' in input_data: # 数据生成的时候用的tp
        # 但是其实这里没有必要
        # import pdb;pdb.set_trace()
        local_command_point = np.array(input_data['my_labels']['target_point'])
    else:
        assert False
    sample['target_point'] = local_command_point
        
    if 'next_command' in input_data:
        command = input_data['next_command'] # command也可以计算，但是没有计算
    elif 'my_labels' in input_data: # 数据生成的时候用的tp
        command = np.array(input_data['my_labels']['command'])
    else:
        assert False

    batch = [sample]
    input_batch = generate_batch(batch)
    
    return input_batch, gt_result, data_route, command, obj_idxs

def get_batch_inp_ndetr(self, label_raw, input_data):
    img = input_data['rgb_front'] # (256, 900, 3) # TODO 改成多相机, TODO bugfix [...,::-1]
    img_shape = img.shape[:2]
    img = torch.tensor(img.transpose(2,0,1)[None,None,...]) # bs,cam,h,w,3
    
    input_batch, gt_pts_bbox, data_route, command, gt_idxs = get_input_batch_plant(label_raw, input_data)
    x, y, _, tp, light = input_batch # (1,2) (1,1) 后面会有stack增加维度
    
    # 下面img_metas中每个是单独一帧
    tp = tp[0].numpy()
    light = light[0].numpy()
    
    # fake gt
    wp = np.zeros((4,2))
    wp[:,0] -= 1.3

    inp = dict(
        img_metas=[dict(
            img_shape = [img_shape],
            topdown = input_data['topdown'],
            hdmap = input_data['hdmap'].transpose(2,0,1),
            lidar2img = [cam2img],
            box_type_3d = box_type_3d,
            gt_idxs = gt_idxs,
            plan = dict(
                wp = wp,
                tp = tp,
                light = light,
                route = data_route,
                command = command,
            )
        )],
        gt_bboxes_3d=[gt_pts_bbox['boxes_3d']],
        gt_labels_3d=[gt_pts_bbox['labels_3d']],
        img=img.to(torch.float32).to('cuda')
    )
    return inp, gt_pts_bbox

global last_time
last_time = time.time()
def time_and_vis(self, inp_dict, gt_pts_bbox, pred_pts_bbox={}):
    frame = self.step // 10
    global last_time
    this_time = time.time()
    print(f"frame: {frame}, use time {(this_time-last_time):.2f}s")
    sys.stdout.flush()
    last_time = this_time

    plan = inp_dict['img_metas'][0]['plan']
    topdown = inp_dict['img_metas'][0]['topdown']
    gt_pts_bbox = dict(
        boxes_3d=gt_pts_bbox.get('boxes_3d', None),
        scores_3d=None,
        labels_3d=None,
        attrs_3d=plan['wp'],
        tp=plan['tp'],
        light=plan['light'],
        command=plan['command'],
        route=plan['route'],
        route_wp=None,
        iscollide=None,
        cam2img=cam2img,
        imgpath=None, # 后面传入img，不需要这个
        topdown=topdown
    )
    out_dir = str(self.save_path / 'vis')
    img = inp_dict['img'][0][0].detach().cpu().numpy().transpose(1,2,0)
    
    show_results(frame, pred_pts_bbox, gt_pts_bbox, out_dir, img=img, in_simu=True) # 进行画图和保存

