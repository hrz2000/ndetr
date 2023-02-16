import os
import json
import math
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from collections import deque
from copy import deepcopy
from PIL import Image, ImageDraw, ImageOps
import time
import mmcv
from carla_agent_files.utils import MapImage, encode_npy_to_pil, PIXELS_PER_METER
import hydra
from hydra import compose, initialize
from carla_agent_files.utils import MapImage, PIXELS_PER_METER
from carla_agent_files.utils import lts_rendering

import cv2
import torch
import numpy as np
import carla

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from carla_agent_files.agent_utils.filter_functions import *
from carla_agent_files.agent_utils import transfuser_utils

from leaderboard.autoagents import autonomous_agent

from carla_agent_files.perception_submissionagent import PerceptionAgent
from training.PlanT.dataset import generate_batch, split_large_BB
from training.PlanT.lit_module import LitHFLM
from training.Perception.config import GlobalConfig

from carla_agent_files.nav_planner import RoutePlanner_new as RoutePlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
import xml.etree.ElementTree as ET

from util.viz_tokens_bev import create_BEV as create_BEV_debug

import pygame
from carla_agent_files.ndetr_follow_plant import get_batch_inp_ndetr, get_bev_boxes, render_BEV
from carla_agent_files.ndetr_follow_plant import time_and_vis
from mmcv import Config, DictAction
from mmdet3d.models import build_model
import sys
sys.path.insert(0,os.path.abspath('./'))
from projects.mmdet3d_plugin import *
from mmcv.runner.checkpoint import load_checkpoint
from training.Perception.config import GlobalConfig

SAVE_PATH = os.environ.get('SAVE_PATH', None)

def get_entry_point():
    return 'PlanTPerceptionAgent'

# SAVE_GIF = os.getenv("SAVE_GIF", 'False').lower() in ('true', '1', 't')

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
        
        self.route_index = route_index
        
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
        args_file = open(os.path.join(path_to_conf_file, f'{self.cfg.model_ckpt_load_path}/log/args.txt'), 'r')
        self.args = json.load(args_file)
        args_file.close()
        self.cfg_agent = OmegaConf.create(self.args)
        self.config = GlobalConfig(setting='eval')

        self.steer_damping = self.config.steer_damping
        self.perception_agent = PerceptionAgent(Path(f'{path_to_conf_file}/{self.cfg.perception_ckpt_load_path}'))
        self.perception_agent.cfg = self.cfg
        
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
        LOAD_CKPT_PATH = f'{path_to_conf_file}/{self.cfg.model_ckpt_load_path}/checkpoints/epoch=0{self.cfg.PlanT_epoch}.ckpt'

        print(f'Loading model from mm')

        # if Path(LOAD_CKPT_PATH).suffix == '.ckpt':
        #     self.net = LitHFLM.load_from_checkpoint(LOAD_CKPT_PATH)
        # else:
        #     raise Exception(f'Unknown model type: {Path(LOAD_CKPT_PATH).suffix}')
        # self.net.eval()
        
        config = self.cfg.mmdet_cfg
        mmdet_cfg = Config.fromfile(config)
        self.use_collide = mmdet_cfg.simu_params.use_collide

        self.ndetr = build_model(
            mmdet_cfg.model,
            train_cfg=mmdet_cfg.get('train_cfg'),
            test_cfg=mmdet_cfg.get('test_cfg'))
        load_checkpoint(self.ndetr, self.cfg.weight, strict=False)
        self.ndetr = self.ndetr.to('cuda')
        self.ndetr.eval()
    
    def _init(self, input_data):
        # super()._init(hd_map)
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
        self.perception_agent._global_plan_world_coord = self._global_plan_world_coord
        self.perception_agent._global_plan = self._global_plan
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
        
        # 因为需要输入hdmap所以需要以下变量
        assert SAVE_PATH != None
        
        route_info = self.route_index + "_" + time.strftime('%Y-%m-%d_%H:%M:%S')
        self.save_path = Path(SAVE_PATH) / route_info
        self.save_path.mkdir(parents=True, exist_ok=False)
        (self.save_path / 'vis').mkdir()

        self.save_lane = True
        self.save_hdmap = True
        
        # Privileged
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
            
        if self.save_lane:
            topology = [x[0] for x in self.hd_map.get_topology()]
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

        if self.save_hdmap:
            # create map for renderer
            map_image = MapImage(self._world, self.hd_map, PIXELS_PER_METER)#5
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
            
    def lateral_shift(self, transform, shift):
        transform.rotation.yaw += 90
        transform.location += shift * transform.get_forward_vector()
        return transform

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
                    {
                        'type': 'sensor.lidar.ray_cast',
                        'x': self.cfg.lidar_pos[0], 
                        'y': self.cfg.lidar_pos[1], 
                        'z': self.cfg.lidar_pos[2],
                        'roll': self.cfg.lidar_rot[0], 
                        'pitch': self.cfg.lidar_rot[1], 
                        'yaw': self.cfg.lidar_rot[2],
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
                'rgb_front': rgb,
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
        
        bev = render_BEV(self)
        bev = np.asarray(bev.squeeze().cpu()) # (15, 300, 300)
        
        topdown = encode_npy_to_pil(bev)
        topdown=np.moveaxis(topdown,0,2)
        hdmap=np.moveaxis(bev[:2],0,2) # 300,300,2
        
        result['topdown'] = topdown
        result['hdmap'] = hdmap
        
        boxes = get_bev_boxes(self)
        result['boxes'] = boxes

        return result


    @torch.no_grad()
    def run_step(self, input_data, timestamp, keep_ids=None):
        
        self.keep_ids = keep_ids

        self.step += 1
        if not self.initialized:
            if 'hd_map' in input_data.keys():
                self._init(input_data['hd_map'])
            else:
                import pdb;pdb.set_trace()
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
            
        tick_data = self.tick(input_data)
        
        # label_raw_gt = [{'class': 'Car', 'extent': [1.5107464790344238, 4.901683330535889, 2.128324270248413], 'position': [-1.3, 0.0, -2.5], 'yaw': 0, 'num_points': -1, 'distance': -1, 'speed': 0.0, 'brake': 0.0, 'id': 99999}]
        # label_raw_gt = get_bev_boxes(self, input_data=None, pos=tick_data['gps']) # 不必要，方便可视化观察，里面需要好多其他东西，索性不获取了
        label_raw_gt = tick_data['boxes']
        
        self.traffic_light_hazard_pred = False
        self.traffic_light_hazard = False
        # # self.perception_agent._vehicle = self._vehicle
        # import pdb;pdb.set_trace()
        label_raw, pred_traffic_light = self.perception_agent.run_step(input_data, tick_data, self.state_log, label_raw_gt, self.traffic_light_hazard, route_gt_map=tick_data['route_wp'])
        # # if self.cfg.perc_traffic_light:
        # self.traffic_light_hazard_pred = pred_traffic_light
        # self.control = self._get_control(label_raw, label_raw_gt, tick_data)
        
        for t in label_raw:
            if t['class'] == 'Route':
                label_raw_gt.append(t)
                
        # import pdb;pdb.set_trace()
        
        self.control = self._get_control(label_raw=None, label_raw_gt=label_raw_gt, input_data=tick_data)
        # label_raw_gt里面没有route、label_raw里面有route
            
        self.perception_agent.control = self.control

        if self.cfg.traffic_light_brake:
            if self.traffic_light_hazard_pred:
                self.control.brake = 1.0
                self.control.throttle = 0.0
                self.control.steer = 0.0
        
        return self.control


    def _get_control(self, label_raw, label_raw_gt, input_data):
        
        gt_velocity = torch.FloatTensor([input_data['speed']]).unsqueeze(0)
    
        # 便于可视化，不使用这些gt信息, 其实没有特别明白
        # input_batch = self.get_input_batch(label_raw=label_raw_gt, label_raw_gt=label_raw_gt, input_data=input_data)
        # x, y, _, tp, light = input_batch
        # _, _, pred_wp, attn_map = self.net(x, y, target_point=tp, light_hazard=light)
        
        inp_dict, _gt_pts_bbox = get_batch_inp_ndetr(self, label_raw_gt, input_data)
        results = self.ndetr.forward_test(**inp_dict) # TODO
        self.save_freq = 10
        if self.step % self.save_freq == 0:
            time_and_vis(self, inp_dict, _gt_pts_bbox, pred_pts_bbox=results[0]['pts_bbox'])
        
        pts_bbox = results[0]['pts_bbox']
        pred_wp = pts_bbox['attrs_3d'] # torch.Size([4, 2])
        if self.use_collide == True: # TODO
            if pts_bbox['iscollide']:
                pred_wp = pred_wp.new_zeros(pred_wp.shape)
                pred_wp[:,0] -= 1.3
                
        is_stuck = False
        if self.cfg.unblock:
            # unblock
            # divide by 2 because we process every second frame
            # 1100 = 55 seconds * 20 Frames per second, we move for 1.5 second = 30 frames to unblock
            if(self.stuck_detector > self.config.stuck_threshold and self.forced_move < self.config.creep_duration):
                print("Detected agent being stuck. Move for frame: ", self.forced_move)
                is_stuck = True
                self.forced_move += 1


        steer, throttle, brake = self.ndetr.control_pid(pred_wp, gt_velocity, is_stuck)

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
            safety_box = deepcopy(input_data['lidar'])
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

        # viz_trigger = (self.step % 20 == 0 and self.cfg.viz)
        # if viz_trigger and self.step > 2:
        #     create_BEV(label_raw, light, tp, pred_wp, label_raw_gt)
        
        # 可视化的预测box格式不一样，暂时不能用 # TODO
        # save_freq = 10
        # if self.step % save_freq == 0:
        #     frame = self.step // save_freq
        #     print(f"frame:{frame}")
        #     mmcv.mkdir_or_exist('./vis')
        #     bev = create_BEV(label_raw, light, tp, pred_wp, label_raw_gt)
        #     bev = np.array(bev)
        #     fv = input_data['rgb']
            
        #     h, w, _ = fv.shape
        #     bh, bw, _ = bev.shape
        #     all_img = np.zeros((h+bh,max(w,bw),3)) # 黑色背景
        #     # all_img = np.full(shape=(h+s,max(w,s),3),fill_value=255) # 白色背景
        #     all_img[:h,:w] = fv
        #     all_img[h:h+bh,(w-bw)//2:(w-bw)//2+bw] = bev
        #     mmcv.imwrite(all_img, f'./vis/{frame:04d}.png')
            
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

    def get_input_batch(self, label_raw, label_raw_gt, input_data):
        sample = {'input': [], 'output': [], 'brake': [], 'waypoints': [], 'target_point': [], 'light': []}

        if self.cfg_agent.model.training.input_ego:
            data = label_raw
            # data_gt = label_raw_gt
        else:
            data = label_raw[1:] # remove first element (ego vehicle)
            # data_gt = label_raw_gt[1:] # remove first element (ego vehicle)

        data_raw_vehicles = data
        data_raw_route = data


        data_car = [[
            1., # type indicator for cars
            float(x['position'][0])-float(label_raw[0]['position'][0]),
            float(x['position'][1])-float(label_raw[0]['position'][1]),
            float(x['yaw'] * 180 / 3.14159265359), # in degrees
            float(x['speed'] * 3.6), # in km/h
            float(x['extent'][2]),
            float(x['extent'][1]),
            ] for x in data_raw_vehicles if x['class'] == 'Car']

        # if we use the far_node as target waypoint we need the route as input
        data_route = [
            [
                2., # type indicator for route
                float(x['position'][0])-float(label_raw[0]['position'][0]),
                float(x['position'][1])-float(label_raw[0]['position'][1]),
                float(x['yaw'] * 180 / 3.14159265359), # in degrees
                float(x['id']),
                float(x['extent'][2]),
                float(x['extent'][1]),
            ] 
            for j, x in enumerate(data_raw_route)
            if x['class'] == 'Route' 
            and float(x['id']) < self.cfg_agent.model.training.max_NextRouteBBs]
        
        # we split route segment slonger than 10m into multiple segments
        # improves generalization
        data_route_split = []
        for route in data_route:
            if route[6] > 10:
                routes = split_large_BB(route, len(data_route_split))
                data_route_split.extend(routes)
            else:
                data_route_split.append(route)

        data_route = data_route_split[:self.cfg_agent.model.training.max_NextRouteBBs]

        assert len(data_route) <= self.cfg_agent.model.training.max_NextRouteBBs, 'Too many routes'

        if self.cfg_agent.model.training.get('remove_velocity', 'None') == 'input':
            for i in range(len(data_car)):
                data_car[i][4] = 0.

        if self.cfg_agent.model.training.get('route_only_wp', False) == True:
            for i in range(len(data_route)):
                data_route[i][3] = 0.
                data_route[i][-2] = 0.
                data_route[i][-1] = 0.

        # filter vehicle and route by attention scores
        # only keep entries which are in self.keep_ids
        if self.keep_ids is not None:
            data_car = [x for i, x in enumerate(data_car) if i in self.keep_ids]
            assert len(data_car) <= len(self.keep_ids), f'{len(data_car)} <= {len(self.keep_ids)}'

        features = data_car + data_route

        sample['input'] = features

        # dummy data
        sample['output'] = features
        sample['light'] = self.traffic_light_hazard_pred
        
        local_command_point = np.array([input_data['target_point_single'][0], input_data['target_point_single'][1]])
        sample['target_point'] = local_command_point

        batch = [sample]
        
        input_batch = generate_batch(batch)
        
        self.data = data_raw_vehicles
        self.data_car = data_car
        self.data_route = data_route
        
        # self.cnt+=1
        
        # if True:
        #     create_BEV_debug(data_car, data_route, True, False, inp='input', cnt=self.cnt, visualize=True)
        
        return input_batch
    
    
    def get_vehicleID_from_attn_scores(self, attn_vector):
        # get ids of all vehicles in detection range
        data_car_ids = [
            float(x['id'])
            for x in self.data if x['class'] == 'Car']

        # get topk indices of attn_vector
        if self.cfg.topk > len(attn_vector):
            topk = len(attn_vector)
        else:
            topk = self.cfg.topk
        
        # get topk vehicles indices
        attn_indices = np.argpartition(attn_vector, -topk)[-topk:]
        
        # get carla vehicles ids of topk vehicles
        keep_vehicle_ids = []
        for indice in attn_indices:
            if indice < len(data_car_ids):
                keep_vehicle_ids.append(data_car_ids[indice])
        
        # if we don't have any detected vehicle we should not have any ids here
        # otherwise we want #topk vehicles
        if len(self.data_car) > 0:
            assert len(keep_vehicle_ids) == topk
        else:
            assert len(keep_vehicle_ids) == 0
            
        return keep_vehicle_ids, attn_indices
    
    
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
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        del self.ndetr
        self.perception_agent.destroy()
        if self.save_lane:
            del self.polygons
        if self.save_hdmap:
            del self.global_map
            del self.renderer

def create_BEV(labels_org, gt_traffic_light_hazard, target_point, pred_wp, label_raw_gt, pix_per_m=5):

    pred_wp = np.array(pred_wp.squeeze())
    s=0
    max_d = 30
    size = int(max_d*pix_per_m*2)
    origin = (size//2, size//2)
    PIXELS_PER_METER = pix_per_m

    
    # color = [(255, 0, 0), (0, 0, 255)]
    color = [(255), (255)]
   
    
    # create black image
    image_0 = Image.new('L', (size, size))
    image_1 = Image.new('L', (size, size))
    image_2 = Image.new('L', (size, size))
    vel_array = np.zeros((size, size))
    draw0 = ImageDraw.Draw(image_0)
    draw1 = ImageDraw.Draw(image_1)
    draw2 = ImageDraw.Draw(image_2)

    draws = [draw0, draw1, draw2]
    imgs = [image_0, image_1, image_2]
    
    for ix, sequence in enumerate([labels_org]):
        for ixx, vehicle in enumerate(sequence):
            x = -vehicle['position'][1]*PIXELS_PER_METER + origin[1]
            y = -vehicle['position'][0]*PIXELS_PER_METER + origin[0]
            yaw = vehicle['yaw']* 180 / 3.14159265359
            extent_x = vehicle['extent'][2]*PIXELS_PER_METER/2
            extent_y = vehicle['extent'][1]*PIXELS_PER_METER/2
            origin_v = (x, y)
            
            if vehicle['class'] == 'Car':
                p1, p2, p3, p4 = get_coords_BB(x, y, yaw-90, extent_x, extent_y)
                if ixx == 0:
                    for ix in range(3):
                        draws[ix].polygon((p1, p2, p3, p4), outline=color[0]) #, fill=color[ix])
                    ix = 0
                else:                
                    draws[ix].polygon((p1, p2, p3, p4), outline=color[ix]) #, fill=color[ix])
                
                if 'speed' in vehicle:
                    vel = vehicle['speed']*3 #/3.6 # in m/s # just for visu
                    endx1, endy1, endx2, endy2 = get_coords(x, y, yaw-90, vel)
                    draws[ix].line((endx1, endy1, endx2, endy2), fill=color[ix], width=2)

            elif vehicle['class'] == 'Route':
                ix = 1
                image = np.array(imgs[ix])
                point = (int(x), int(y))
                cv2.circle(image, point, radius=3, color=color[0], thickness=3)
                p1, p2, p3, p4 = get_coords_BB(x, y, yaw-90, extent_x, extent_y)
                draws[ix].polygon((p1, p2, p3, p4), outline=color[ix]) #, fill=color[ix])
                imgs[ix] = Image.fromarray(image)
                
                                
    for ix, sequence in enumerate([label_raw_gt]):
        for ixx, vehicle in enumerate(sequence):
            x = -vehicle['position'][1]*PIXELS_PER_METER + origin[1]
            y = -vehicle['position'][0]*PIXELS_PER_METER + origin[0]
            yaw = vehicle['yaw']* 180 / 3.14159265359
            extent_x = vehicle['extent'][2]*PIXELS_PER_METER/2
            extent_y = vehicle['extent'][1]*PIXELS_PER_METER/2
            origin_v = (x, y)
            
            if vehicle['class'] == 'Car':
                p1, p2, p3, p4 = get_coords_BB(x, y, yaw-90, extent_x, extent_y)
                if ixx == 0:
                    pass
                else:                
                    draws[2].polygon((p1, p2, p3, p4), outline=color[0]) #, fill=color[ix])
                
                if 'speed' in vehicle:
                    vel = vehicle['speed']*3 #/3.6 # in m/s # just for visu
                    endx1, endy1, endx2, endy2 = get_coords(x, y, yaw-90, vel)
                    draws[2].line((endx1, endy1, endx2, endy2), fill=color[0], width=2)

            elif vehicle['class'] == 'Route':
                ix = 2
                image = np.array(imgs[ix])
                point = (int(x), int(y))
                cv2.circle(image, point, radius=3, color=color[0], thickness=1)
                p1, p2, p3, p4 = get_coords_BB(x, y, yaw-90, extent_x, extent_y)
                draws[ix].polygon((p1, p2, p3, p4), outline=color[0]) #, fill=color[ix])
                imgs[ix] = Image.fromarray(image)
                
    
                
    for wp in pred_wp:
        x = wp[1]*PIXELS_PER_METER + origin[1]
        y = -wp[0]*PIXELS_PER_METER + origin[0]
        image = np.array(imgs[2])
        point = (int(x), int(y))
        cv2.circle(image, point, radius=2, color=255, thickness=2)
        imgs[2] = Image.fromarray(image)
          
    image = np.array(imgs[0])
    image1 = np.array(imgs[1])
    image2 = np.array(imgs[2])
    x = target_point[0][1]*PIXELS_PER_METER + origin[1]
    y = -(target_point[0][0])*PIXELS_PER_METER + origin[0]  
    point = (int(x), int(y))
    cv2.circle(image, point, radius=2, color=color[0], thickness=2)
    cv2.circle(image1, point, radius=2, color=color[0], thickness=2)
    cv2.circle(image2, point, radius=2, color=color[0], thickness=2)
    imgs[0] = Image.fromarray(image)
    imgs[1] = Image.fromarray(image1)
    imgs[2] = Image.fromarray(image2)
    
    images = [np.asarray(img) for img in imgs]
    image = np.stack([images[0], images[2], images[1]], axis=-1)
    BEV = image

    img_final = Image.fromarray(image.astype(np.uint8))
    if gt_traffic_light_hazard:
        color = 'red'
    else:
        color = 'green'
    img_final = ImageOps.expand(img_final, border=5, fill=color)
    
    
    
    ## add rgb image and lidar
    # all_images = np.concatenate((images_lidar, np.array(img_final)), axis=1)
    # all_images = np.concatenate((rgb_image, all_images), axis=0)
    all_images = img_final
    
    # Path(f'bev_viz1').mkdir(parents=True, exist_ok=True)
    # all_images.save(f'bev_viz1/{time.time()}_{s}.png')

    return all_images


def get_coords(x, y, angle, vel):
    length = vel
    endx2 = x + length * math.cos(math.radians(angle))
    endy2 = y + length * math.sin(math.radians(angle))

    return x, y, endx2, endy2  

def get_coords_BB(x, y, angle, extent_x, extent_y):
    endx1 = x - extent_x * math.sin(math.radians(angle)) - extent_y * math.cos(math.radians(angle))
    endy1 = y + extent_x * math.cos(math.radians(angle)) - extent_y * math.sin(math.radians(angle))

    endx2 = x + extent_x * math.sin(math.radians(angle)) - extent_y * math.cos(math.radians(angle))
    endy2 = y - extent_x * math.cos(math.radians(angle)) - extent_y * math.sin(math.radians(angle))

    endx3 = x + extent_x * math.sin(math.radians(angle)) + extent_y * math.cos(math.radians(angle))
    endy3 = y - extent_x * math.cos(math.radians(angle)) + extent_y * math.sin(math.radians(angle))

    endx4 = x - extent_x * math.sin(math.radians(angle)) + extent_y * math.cos(math.radians(angle))
    endy4 = y + extent_x * math.cos(math.radians(angle)) + extent_y * math.sin(math.radians(angle))

    return (endx1, endy1), (endx2, endy2), (endx3, endy3), (endx4, endy4)