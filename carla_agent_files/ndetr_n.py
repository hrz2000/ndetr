import json
import os
import random
import cv2
from copy import deepcopy
from pathlib import Path

import torch
import numpy as np
from rdp import rdp
import sys

from carla_agent_files.autopilot import AutoPilot
# from scenario_logger import ScenarioLogger
import carla
from carla_agent_files.utils import MapImage, PIXELS_PER_METER
from carla_agent_files.utils import lts_rendering
import pygame
from carla_agent_files.ndetr_follow_plant import get_batch_inp_ndetr, get_bev_boxes, render_BEV
from carla_agent_files.ndetr_follow_plant import time_and_vis
from mmcv import Config, DictAction
from mmdet3d.models import build_model
sys.path.insert(0,os.path.abspath('./'))
from projects.mmdet3d_plugin import *
from mmcv.runner.checkpoint import load_checkpoint
from training.Perception.config import GlobalConfig

SHUFFLE_WEATHER = int(os.environ.get('SHUFFLE_WEATHER'))

WEATHERS = {
		'Clear': carla.WeatherParameters.ClearNoon,
		'Cloudy': carla.WeatherParameters.CloudySunset,
		'Wet': carla.WeatherParameters.WetSunset,
		'MidRain': carla.WeatherParameters.MidRainSunset,
		'WetCloudy': carla.WeatherParameters.WetCloudySunset,
		'HardRain': carla.WeatherParameters.HardRainNoon,
		'SoftRain': carla.WeatherParameters.SoftRainSunset,
}

azimuths = [45.0 * i for i in range(8)]

daytimes = {
	'Night': -80.0,
	'Twilight': 0.0,
	'Dawn': 5.0,
	'Sunset': 15.0,
	'Morning': 35.0,
	'Noon': 75.0,
}

WEATHERS_IDS = list(WEATHERS)


def get_entry_point():
    # import pdb;pdb.set_trace()
    return 'DataAgent'


class DataAgent(AutoPilot):
    def setup(self, path_to_conf_file, route_index=None, cfg=None, exec_or_inter=None):
        super().setup(path_to_conf_file, route_index, cfg, exec_or_inter)

        # self.args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.all_time = True
        self.pos_dict = {}
        self.cfg = cfg # 是experiments
        self.save_vis = self.cfg.save_vis
        self.save_lane = self.cfg.save_lane
        self.save_hdmap = self.cfg.save_hdmap
        
        self.map_precision = 10.0 # meters per point
        self.rdp_epsilon = 0.5 # epsilon for route shortening

        # radius in which other actors/map elements are considered
        # distance is from the center of the ego-vehicle and measured in 3D space
        self.max_actor_distance = self.detection_radius # copy from expert
        self.max_light_distance = self.light_radius # copy from expert
        self.max_route_distance = 30.0
        self.max_map_element_distance = 30.0

        # if self.log_path is not None:
        #     self.log_path = Path(self.log_path) / route_index
        #     Path(self.log_path).mkdir(parents=True, exist_ok=True) 
        
        # self.scenario_logger = ScenarioLogger(
        #     save_path=self.log_path, 
        #     route_index=self.route_index,
        #     logging_freq=self.save_freq,
        #     log_only=False,
        #     route_only=False, # with vehicles and lights
        #     roi = self.detection_radius+10,
        # )
        self.stuck_detector = 0
        self.forced_move = 0
        self.use_lidar_safe_check = False
        self.config = GlobalConfig(setting='eval')
        
        assert self.save_path!=None
        if self.save_path is not None:
            (self.save_path / 'boxes').mkdir()
            
            if self.cfg.SAVE_SENSORS:
                # (self.save_path / 'rgb').mkdir()
                (self.save_path / 'topdown').mkdir()
                (self.save_path / 'hdmap0').mkdir()
                (self.save_path / 'hdmap1').mkdir()
                # (self.save_path / 'rgb_augmented').mkdir()
                # (self.save_path / 'lidar').mkdir()

        ###
        config = self.cfg.mmdet_cfg
        mmdet_cfg = Config.fromfile(config)
        self.use_collide = mmdet_cfg.simu_params.use_collide

        self.net = build_model(
            mmdet_cfg.model,
            train_cfg=mmdet_cfg.get('train_cfg'),
            test_cfg=mmdet_cfg.get('test_cfg'))
        load_checkpoint(self.net, self.cfg.weight, strict=False)
        self.net = self.net.to('cuda')

        self.net.eval()
        
    def _init(self, hd_map):
        super()._init(hd_map)

        # if self.scenario_logger:
        #     from srunner.scenariomanager.carla_data_provider import CarlaDataProvider # privileged
        #     self._vehicle = CarlaDataProvider.get_hero_actor()
        #     self.scenario_logger.ego_vehicle = self._vehicle
        #     self.scenario_logger.world = self._vehicle.get_world()

        if self.save_lane:
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

        if self.save_hdmap:
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
        result = super().sensors()
        if self.save_path is not None and self.cfg.SAVE_SENSORS:
            result += [{
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
            #     'type': 'sensor.camera.rgb',
            #     'x': self.cfg.camera_pos[0],
            #     'y': self.cfg.camera_pos[1],
            #     'z': self.cfg.camera_pos[2],
            #     'roll': self.cfg.camera_rot_0[0],
            #     'pitch': self.cfg.camera_rot_0[1],
            #     'yaw': self.cfg.camera_rot_0[2],
            #     'width': self.cfg.camera_width,
            #     'height': self.cfg.camera_height,
            #     'fov': self.cfg.camera_fov_data_collection,
            #     'id': 'rgb_augmented'
            # }
            ]

            # result.append({
            #     'type': 'sensor.lidar.ray_cast',
            #     'x': self.cfg.lidar_pos[0],
            #     'y': self.cfg.lidar_pos[1],
            #     'z': self.cfg.lidar_pos[2],
            #     'roll': self.cfg.lidar_rot[0],
            #     'pitch': self.cfg.lidar_rot[1],
            #     'yaw': self.cfg.lidar_rot[2],
            #     'rotation_frequency': self.cfg.lidar_rotation_frequency,
            #     'points_per_second': self.cfg.lidar_points_per_second,
            #     'id': 'lidar'
            # })


        return result

    def tick_ndetr(self, input_data):
        result = super().tick(input_data)

        if self.save_path is not None:
            boxes = get_bev_boxes(self)
            assert self.cfg.SAVE_SENSORS==True
            if self.cfg.SAVE_SENSORS:
                rgb = []
                for pos in ['front']:
                    rgb_cam = 'rgb_' + pos

                    rgb.append(input_data[rgb_cam][1][:, :, :3])

                rgb = np.concatenate(rgb, axis=1)

                # rgb_augmented = input_data['rgb_augmented'][1][:, :, :3]

                # lidar = input_data['lidar']
            else:
                rgb = None
                rgb_augmented = None
                lidar = None

            
        else:
            rgb = None
            rgb_augmented = None
            boxes = None
            lidar = None


        result.update({'rgb': rgb,
                        # 'rgb_augmented': rgb_augmented,
                        'boxes': boxes,
                        # 'lidar': lidar
                        })

        if 'my_labels' in input_data:
            result['my_labels'] = input_data['my_labels']

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp, sensors=None):
        # Must be called before run_step, so that the correct augmentation shift is
        # Saved
        control_gt = super().run_step(input_data, timestamp)
        assert 'my_labels' in input_data

        self.control = self._get_control_ndetr(input_data)

        if self.traffic_light_hazard:
            self.control.brake = 1.0
            self.control.throttle = 0.0
            self.control.steer = 0.0
            
        return self.control

    def _get_control_ndetr(self, input_data):# dict_keys(['gps', 'imu', 'hd_map', 'speed', 'my_labels'])
        tick_data = self.tick_ndetr(input_data) # dict_keys(['gps', 'speed', 'compass', 'rgb', 'boxes', 'my_labels'])
        label_raw = get_bev_boxes(self, input_data=input_data, pos=tick_data['gps'])
        label_raw_pred = None
        gt_velocity = torch.FloatTensor([tick_data['speed']]).unsqueeze(0)
        
        inp_dict, gt_pts_bbox = get_batch_inp_ndetr(self, label_raw, tick_data)
        with torch.no_grad():
            results = self.net.forward_test(**inp_dict)
            
        if self.step % self.save_freq == 0: # 10
            time_and_vis(self, inp_dict, gt_pts_bbox, pred_pts_bbox=results[0]['pts_bbox'], save_hdmapx=self.save_hdmap, save_vis=self.save_vis)
            # 其他数据在data_agent里被保存
                    
        pts_bbox = results[0]['pts_bbox']
        pred_wp = pts_bbox['attrs_3d'] # torch.Size([4, 2])
        if self.use_collide == True:
            if pts_bbox['iscollide']:
                pred_wp = pred_wp.new_zeros(pred_wp.shape)
                pred_wp[:,0] -= 1.3

        is_stuck = False
        if self.cfg.unblock: # 后面开启试试
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
                
            if self.use_lidar_safe_check:
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

    def augment_camera(self, sensors):
        for sensor in sensors:
            if 'rgb_augmented' in sensor[0]:
                augmentation_translation = np.random.uniform(low=self.cfg.camera_translation_augmentation_min,
                                                            high=self.cfg.camera_translation_augmentation_max)
                augmentation_rotation = np.random.uniform(low=self.cfg.camera_rotation_augmentation_min,
                                                        high=self.cfg.camera_rotation_augmentation_max)
                self.augmentation_translation.append(augmentation_translation)
                self.augmentation_rotation.append(augmentation_rotation)
                camera_pos_augmented = carla.Location(x=self.cfg.camera_pos[0],
                                                    y=self.cfg.camera_pos[1] + augmentation_translation,
                                                    z=self.cfg.camera_pos[2])

                camera_rot_augmented = carla.Rotation(pitch=self.cfg.camera_rot_0[0],
                                                    yaw=self.cfg.camera_rot_0[1] + augmentation_rotation,
                                                    roll=self.cfg.camera_rot_0[2])

                camera_augmented_transform = carla.Transform(camera_pos_augmented, camera_rot_augmented)

                sensor[1].set_transform(camera_augmented_transform)

    def shuffle_weather(self):
        # change weather for visual diversity
        index = random.choice(range(len(WEATHERS)))
        dtime, altitude = random.choice(list(daytimes.items()))
        altitude = np.random.normal(altitude, 10)
        self.weather_id = WEATHERS_IDS[index] + dtime

        weather = WEATHERS[WEATHERS_IDS[index]]
        weather.sun_altitude_angle = altitude
        weather.sun_azimuth_angle = np.random.choice(azimuths)
        self._world.set_weather(weather)

        # night mode
        vehicles = self._world.get_actors().filter('*vehicle*')
        if weather.sun_altitude_angle < 0.0:
            for vehicle in vehicles:
                vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))
        else:
            for vehicle in vehicles:
                vehicle.set_light_state(carla.VehicleLightState.NONE)


    # def save_sensors(self, tick_data):
    #     frame = self.step // self.save_freq
        
    #     if self.cfg.SAVE_SENSORS:
    #         # CV2 uses BGR internally so we need to swap the image channels before saving.
    #         cv2.imwrite(str(self.save_path / 'rgb' / (f'{frame:04}.png')), tick_data['rgb'])
    #         # cv2.imwrite(str(self.save_path / 'rgb_augmented' / (f'{frame:04}.png')), tick_data['rgb_augmented'])
    #         # np.save(self.save_path / 'lidar' / ('%04d.npy' % frame), tick_data['lidar'], allow_pickle=True)
    #         inp, gt_result = get_batch_inp_ndetr(self, None, tick_data['boxes'], tick_data)
    #         save_bev(self,
    #             gt_result=gt_result, 
    #             pred_result=None,
    #             plan=inp['img_metas'][0]['plan'],
    #             inp_img=inp['img'][0][0].detach().cpu().numpy().transpose(1,2,0),
    #             save_hdmapx=True,
    #             save_vis=True)

    #     self.save_labels(self.save_path / 'boxes' / ('%04d.json' % frame), tick_data['boxes'])
        
    def save_labels(self, filename, result):
        with open(filename, 'w') as f:
            json.dump(result, f, indent=4)
        return

    def save_points(self, filename, points):
        points_to_save = deepcopy(points[1])
        points_to_save[:, 1] = -points_to_save[:, 1]
        np.save(filename, points_to_save)
        return
    
    def destroy(self):
        del self.net
        if self.save_lane:
            del self.polygons
        if self.save_hdmap:
            del self.global_map
            del self.renderer
        # if self.scenario_logger:
        #     self.scenario_logger.dump_to_json()
        #     del self.scenario_logger
    
    def lateral_shift(self, transform, shift):
        transform.rotation.yaw += 90
        transform.location += shift * transform.get_forward_vector()
        return transform

    def get_points_in_bbox(self, ego_matrix, vehicle_matrix, dx, lidar):
        # inverse transform lidar to 
        Tr_lidar_2_ego = self.get_lidar_to_vehicle_transform()
        
        # construct transform from lidar to vehicle
        Tr_lidar_2_vehicle = np.linalg.inv(vehicle_matrix) @ ego_matrix @ Tr_lidar_2_ego

        # transform lidar to vehicle coordinate
        lidar_vehicle = Tr_lidar_2_vehicle[:3, :3] @ lidar[1][:, :3].T + Tr_lidar_2_vehicle[:3, 3:]

        # check points in bbox
        x, y, z = dx / 2.
        # why should we use swap?
        x, y = y, x
        num_points = ((lidar_vehicle[0] < x) & (lidar_vehicle[0] > -x) & 
                      (lidar_vehicle[1] < y) & (lidar_vehicle[1] > -y) & 
                      (lidar_vehicle[2] < z) & (lidar_vehicle[2] > -z)).sum()
        return num_points

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

    def get_lidar_to_vehicle_transform(self):
        # yaw = -90
        rot = np.array([[0, 1, 0],
                        [-1, 0, 0],
                        [0, 0, 1]], dtype=np.float32)
        T = np.eye(4)

        T[0, 3] = 1.3
        T[1, 3] = 0.0
        T[2, 3] = 2.5
        T[:3, :3] = rot
        return T

        
    def get_vehicle_to_lidar_transform(self):
        return np.linalg.inv(self.get_lidar_to_vehicle_transform())

    def get_image_to_vehicle_transform(self):
        # yaw = 0.0 as rot is Identity
        T = np.eye(4)
        T[0, 3] = 1.3
        T[1, 3] = 0.0
        T[2, 3] = 2.3

        # rot is from vehicle to image
        rot = np.array([[0, -1, 0],
                        [0, 0, -1],
                        [1, 0, 0]], dtype=np.float32)
        
        # so we need a transpose here
        T[:3, :3] = rot.T
        return T

    def get_vehicle_to_image_transform(self):
        return np.linalg.inv(self.get_image_to_vehicle_transform())

    def get_lidar_to_image_transform(self):
        Tr_lidar_to_vehicle = self.get_lidar_to_vehicle_transform()
        Tr_image_to_vehicle = self.get_image_to_vehicle_transform()
        T_lidar_to_image = np.linalg.inv(Tr_image_to_vehicle) @ Tr_lidar_to_vehicle
        return T_lidar_to_image