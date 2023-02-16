import os
import json
import time
from pathlib import Path
from tkinter import N
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageOps
import pickle
import cv2
import torch
import numpy as np
import carla

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from carla_agent_files.agent_utils.filter_functions import *
from carla_agent_files.agent_utils.coordinate_utils import preprocess_compass, inverse_conversion_2d
from carla_agent_files.agent_utils.explainability_utils import *

from carla_agent_files.data_agent_boxes import DataAgent
from training.PlanT.dataset import generate_batch, split_large_BB
from training.PlanT.lit_module import LitHFLM

from carla_agent_files.nav_planner import extrapolate_waypoint_route
from carla_agent_files.nav_planner import RoutePlanner_new as RoutePlanner
from carla_agent_files.scenario_logger import ScenarioLogger

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

import hydra
from hydra import compose, initialize


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

def get_entry_point():
    return 'PlanTAgent'

SAVE_GIF = os.getenv("SAVE_GIF", 'False').lower() in ('true', '1', 't')


class PlanTAgent(DataAgent):
    def setup(self, path_to_conf_file, route_index=None, cfg=None, exec_or_inter=None):
        self.cfg = cfg
        self.exec_or_inter = exec_or_inter

        # first args than super setup is important!
        args_file = open(os.path.join(path_to_conf_file, 'args.txt'), 'r')
        self.args = json.load(args_file)
        args_file.close()
        self.cfg_agent = OmegaConf.create(self.args)

        super().setup(path_to_conf_file, route_index, cfg, exec_or_inter)

        print(f'Saving gif: {SAVE_GIF}')

        
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
                    dt=1/self.frame_rate,
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
        self.state_log = deque(maxlen=2)


        # exec_or_inter is used for the interpretability metric
        # exec is the model that executes the actions in carla
        # inter is the model that obtains attention scores and a ranking of the vehicles importance
        if exec_or_inter is not None:
            if exec_or_inter == 'exec':
                LOAD_CKPT_PATH = cfg.exec_model_ckpt_load_path
            elif exec_or_inter == 'inter':
                LOAD_CKPT_PATH = cfg.inter_model_ckpt_load_path
        else:
            LOAD_CKPT_PATH = cfg.model_ckpt_load_path

        print(f'Loading model from {LOAD_CKPT_PATH}')

        if Path(LOAD_CKPT_PATH).suffix == '.ckpt':
            self.net = LitHFLM.load_from_checkpoint(LOAD_CKPT_PATH)
        else:
            raise Exception(f'Unknown model type: {Path(LOAD_CKPT_PATH).suffix}')
        self.net.eval()
        self.scenario_logger = False

        if self.log_path is not None:
            self.log_path = Path(self.log_path) / route_index
            Path(self.log_path).mkdir(parents=True, exist_ok=True)   
                 
            self.scenario_logger = ScenarioLogger(
                save_path=self.log_path, 
                route_index=self.route_index,
                logging_freq=self.save_freq,
                log_only=False,
                route_only=False, # with vehicles and lights
                roi = self.detection_radius+10,
            )

        self.config = GlobalConfig(setting='eval')
        self.steer_damping = self.config.steer_damping
        self.perception_agent = PerceptionAgent(Path(f'{path_to_conf_file}/{self.cfg.perception_ckpt_load_path}'))
        self.perception_agent.cfg = self.cfg

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

        # para
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

        # for lane
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

        # for hdmap
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()
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


    def sensors(self):
        # result = super().sensors()
        result = [{
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
            },{
                'type': 'sensor.camera.rgb',
                'x': self.cfg.camera_pos[0],
                'y': self.cfg.camera_pos[1],
                'z': self.cfg.camera_pos[2],
                'roll': self.cfg.camera_rot_0[0],
                'pitch': self.cfg.camera_rot_0[1],
                'yaw': np.pi*40/180,
                'width': self.cfg.camera_width,
                'height': self.cfg.camera_height,
                'fov': self.cfg.camera_fov_data_collection,
                'id': 'rgb_left'
            },{
                'type': 'sensor.camera.rgb',
                'x': self.cfg.camera_pos[0],
                'y': self.cfg.camera_pos[1],
                'z': self.cfg.camera_pos[2],
                'roll': self.cfg.camera_rot_0[0],
                'pitch': self.cfg.camera_rot_0[1],
                'yaw': -np.pi*40/180,
                'width': self.cfg.camera_width,
                'height': self.cfg.camera_height,
                'fov': self.cfg.camera_fov_data_collection,
                'id': 'rgb_right'
            },{
                    'type': 'sensor.opendrive_map',
                    'reading_frequency': 1e-6,
                    'id': 'hd_map'
                    },
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                    }]
        return result


    def tick(self, input_data):
        # result = super().tick(input_data)
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        if (math.isnan(compass) == True): # simulation bug
            compass = 0.0

        result = {
                'gps': gps,
                'speed': speed,
                'compass': compass,
                }
        
        rgb_front = []
        rgb_front.append(cv2.cvtColor(input_data['rgb_front'][1][:, :, :3], cv2.COLOR_BGR2RGB))
        rgb_front = np.concatenate(rgb_front, axis=1)

        rgb_left = []
        rgb_left.append(cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB))
        rgb_left = np.concatenate(rgb_left, axis=1)

        rgb_right = []
        rgb_right.append(cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB))
        rgb_right = np.concatenate(rgb_right, axis=1)
        
        result['rgb_front'] = rgb_front
        result['rgb_left'] = rgb_left
        result['rgb_right'] = rgb_right

        pos = self._route_planner.convert_gps_to_carla(input_data['gps'][1][:2])
        speed = input_data['speed'][1]['speed']
        compass = preprocess_compass(input_data['imu'][1][-1])

        
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

        ego_target_point = inverse_conversion_2d(target_point, result['gps'], compass)
        result['target_point'] = tuple(ego_target_point)

        # if SAVE_GIF == True and (self.exec_or_inter == 'inter'):
        #     result['rgb_back'] = input_data['rgb_back']
        #     result['sem_back'] = input_data['sem_back']
            
        if self.scenario_logger:
            waypoint_route = self._waypoint_planner.run_step(filtered_state[0:2])
            waypoint_route = extrapolate_waypoint_route(waypoint_route,
                                                        10)
            route = np.array([[node[0][1], -node[0][0]] for node in waypoint_route]) \
                [:10]
            # Logging
            self.scenario_logger.log_step(route)

        bev = render_BEV(self) #torch.Size([1, 15, 500, 500])
        bev_large = np.asarray(bev.squeeze().cpu())
        topdown = encode_npy_to_pil(bev_large)
        result['topdown'] = topdown
        
        hdmap=np.moveaxis(bev_large[:2],0,2) # 300,300,2
        result['hdmap'] = hdmap
    
        return result

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

    @torch.no_grad()
    def run_step(self, input_data, timestamp, sensors=None,  keep_ids=None):
        
        self.keep_ids = keep_ids

        self.step += 1
        if not self.initialized:
            if ('hd_map' in input_data.keys()):
                self._init(input_data['hd_map'])
            else:
                self.control = carla.VehicleControl()
                self.control.steer = 0.0
                self.control.throttle = 0.0
                self.control.brake = 1.0
                if self.exec_or_inter == 'inter':
                    return [], None
                return self.control

        # needed for traffic_light_hazard
        _ = super()._get_brake(stop_sign_hazard=0, vehicle_hazard=0, walker_hazard=0)
        tick_data = self.tick(input_data)
        label_raw = super().get_bev_boxes(input_data=input_data, pos=tick_data['gps'])
        
        if self.exec_or_inter == 'inter':
            keep_vehicle_ids = self._get_control(label_raw, tick_data)
            # print(f'plant: {keep_vehicle_ids}')
            
            return keep_vehicle_ids
        elif self.exec_or_inter == 'exec' or self.exec_or_inter is None:
            self.control = self._get_control(label_raw, tick_data)
            
        
        inital_frames_delay = 40
        if self.step < inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
            
        return self.control


    def _get_control(self, label_raw, input_data):
        
        gt_velocity = torch.FloatTensor([input_data['speed']]).unsqueeze(0)
        input_batch = self.get_input_batch(label_raw, input_data) ## right
        x, y, _, tp, light = input_batch
    
        _, _, pred_wp, attn_map = self.net(x, y, target_point=tp, light_hazard=light)

        steer, throttle, brake = self.net.model.control_pid(pred_wp[:1], gt_velocity)

        if brake < 0.05: brake = 0.0
        if throttle > brake: brake = 0.0

        if brake:
            steer *= self.steer_damping

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        # viz_trigger = ((self.step % 20 == 0) and self.cfg.viz)
        # if viz_trigger and self.step > 2:
        #     create_BEV(label_raw, light, tp, pred_wp)
        

        if self.exec_or_inter == 'inter':
            attn_vector = get_attn_norm_vehicles(self.cfg.attention_score, self.data_car, attn_map)
            keep_vehicle_ids, attn_indices, keep_vehicle_attn = get_vehicleID_from_attn_scores(self.data, self.data_car, self.cfg.topk, attn_vector)
            if SAVE_GIF == True and (self.exec_or_inter == 'inter'):
                draw_attention_bb_in_carla(self._world, keep_vehicle_ids, keep_vehicle_attn, self.frame_rate_sim)
                if self.step % 1 == 0:
                    get_masked_viz_3rd_person(self.save_path_org, self.save_path_mask, self.step, input_data)
            
            return keep_vehicle_ids, attn_indices
    
        if self.step % 10 == 0:
            frame = self.step // 10
            print(frame)

            cv2.imwrite(str(self.save_path / 'rgb_front' / (f'{frame:04}.png')), input_data['rgb_front'])
            cv2.imwrite(str(self.save_path / 'rgb_left' / (f'{frame:04}.png')), input_data['rgb_left'])
            cv2.imwrite(str(self.save_path / 'rgb_right' / (f'{frame:04}.png')), input_data['rgb_right'])
            
            with open(self.save_path/'measurements'/f'{frame:04}.json', 'w', encoding='utf-8') as f:
                json.dump(input_data['measurements'], f, indent=4)
                
            with open(self.save_path / 'boxes' / ('%04d.json' % frame), 'w') as f:
                json.dump(input_data['boxes'], f, indent=4)
                
            with open(self.save_path / 'hdmap' / ('%04d.json' % frame),'wb') as f:
                pickle.dump(input_data['hdmap'], f , protocol=pickle.HIGHEST_PROTOCOL)

        return control
        
    def save_labels(self, filename, result):
        with open(filename, 'w') as f:
            json.dump(result, f, indent=4)
        return
    
    def get_input_batch(self, label_raw, input_data):
        sample = {'input': [], 'output': [], 'brake': [], 'waypoints': [], 'target_point': [], 'light': []}

        if self.cfg_agent.model.training.input_ego:
            data = label_raw
        else:
            data = label_raw[1:] # remove first element (ego vehicle)

        data_car = [[
            1., # type indicator for cars
            float(x['position'][0])-float(label_raw[0]['position'][0]),
            float(x['position'][1])-float(label_raw[0]['position'][1]),
            float(x['yaw'] * 180 / 3.14159265359), # in degrees
            float(x['speed'] * 3.6), # in km/h
            float(x['extent'][2]),
            float(x['extent'][1]),
            ] for x in data if x['class'] == 'Car'] # and ((self.cfg_agent.model.training.remove_back and float(x['position'][0])-float(label_raw[0]['position'][0]) >= 0) or not self.cfg_agent.model.training.remove_back)]

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
            for j, x in enumerate(data)
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
        sample['light'] = self.traffic_light_hazard

        local_command_point = np.array([input_data['target_point'][0], input_data['target_point'][1]])
        sample['target_point'] = local_command_point

        batch = [sample]
        
        input_batch = generate_batch(batch)
        
        self.data = data
        self.data_car = data_car
        self.data_route = data_route
        
        return input_batch
    
    
    def destroy(self):
        super().destroy()
        if self.scenario_logger:
            self.scenario_logger.dump_to_json()
            del self.scenario_logger
            
        if SAVE_GIF == True and (self.exec_or_inter == 'inter'):
            self.save_path_mask_vid = f'viz_vid/masked'
            self.save_path_org_vid = f'viz_vid/org'
            Path(self.save_path_mask_vid).mkdir(parents=True, exist_ok=True)
            Path(self.save_path_org_vid).mkdir(parents=True, exist_ok=True)
            out_name_mask = f"{self.save_path_mask_vid}/{self.route_index}.mp4"
            out_name_org = f"{self.save_path_org_vid}/{self.route_index}.mp4"
            cmd_mask = f"ffmpeg -r 25 -i {self.save_path_mask}/%d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {out_name_mask}"
            cmd_org = f"ffmpeg -r 25 -i {self.save_path_org}/%d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {out_name_org}"
            print(cmd_mask)
            os.system(cmd_mask)
            print(cmd_org)
            os.system(cmd_org)
            
            # delete the images
            os.system(f"rm -rf {Path(self.save_path_mask).parent}")
            
        del self.net
        


def create_BEV(labels_org, gt_traffic_light_hazard, target_point, pred_wp, pix_per_m=5):

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
               
        # features = rearrange(features, '(vehicle features) -> vehicle features', features=4)
        for ixx, vehicle in enumerate(sequence):
            # draw vehicle
            # if vehicle['class'] != 'Car':
            #     continue
            
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
    
    Path(f'bev_viz').mkdir(parents=True, exist_ok=True)
    all_images.save(f'bev_viz/{time.time()}_{s}.png')

    # return BEV


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

