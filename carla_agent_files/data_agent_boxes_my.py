import json
import os
import random
import cv2
from copy import deepcopy
from pathlib import Path
import pickle
import torch
import numpy as np
from rdp import rdp
from carla_agent_files.ndetr_follow_plant import render_BEV

from carla_agent_files.autopilot import AutoPilot
# from scenario_logger import ScenarioLogger
import carla
from carla_agent_files.utils import MapImage, encode_npy_to_pil, PIXELS_PER_METER
from carla_agent_files.utils import lts_rendering
import pygame
from carla_agent_files.ndetr_follow_plant import get_batch_inp_ndetr, get_bev_boxes
from carla_agent_files.ndetr_follow_plant import time_and_vis

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
    return 'DataAgent'


class DataAgent(AutoPilot):
    def setup(self, path_to_conf_file, route_index=None, cfg=None, exec_or_inter=None):
        super().setup(path_to_conf_file, route_index, cfg, exec_or_inter)

        # self.args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cfg = cfg
        self.save_lane = True
        self.save_hdmap = True

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

        if self.save_path is not None:
            (self.save_path / 'boxes').mkdir()
            
            if self.cfg.SAVE_SENSORS:
                (self.save_path / 'rgb').mkdir()
                (self.save_path / 'rgb_front').mkdir()
                (self.save_path / 'rgb_left').mkdir()
                (self.save_path / 'rgb_right').mkdir()
                (self.save_path / 'topdown').mkdir()
                (self.save_path / 'hdmap').mkdir()
                # (self.save_path / 'rgb_augmented').mkdir()
                # (self.save_path / 'lidar').mkdir()
        
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
            }, {
                'type': 'sensor.camera.rgb',
                'x': self.cfg.camera_pos[0],
                'y': self.cfg.camera_pos[1],
                'z': self.cfg.camera_pos[2],
                'roll': self.cfg.camera_rot_0[0],
                'pitch': self.cfg.camera_rot_0[1],
                # 'yaw': np.pi*40/180,
                'yaw': 40,
                'width': self.cfg.camera_width,
                'height': self.cfg.camera_height,
                'fov': self.cfg.camera_fov_data_collection,
                'id': 'rgb_right'
            },{
                'type': 'sensor.camera.rgb',
                'x': self.cfg.camera_pos[0],
                'y': self.cfg.camera_pos[1],
                'z': self.cfg.camera_pos[2],
                'roll': self.cfg.camera_rot_0[0],
                'pitch': self.cfg.camera_rot_0[1],
                'yaw': -40,
                'width': self.cfg.camera_width,
                'height': self.cfg.camera_height,
                'fov': self.cfg.camera_fov_data_collection,
                'id': 'rgb_left'
            }
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

    def tick(self, input_data): # dict_keys(['gps', 'imu', 'hd_map', 'speed', 'rgb_front', 'my_labels']) dict_keys(['gps', 'imu', 'speed', 'hd_map', 'rgb_left', 'rgb_right', 'rgb_front', 'my_labels']) my_labels说的是调用了autopilot的measn信息 TODO
        result = super().tick(input_data)

        if self.save_path is not None:
            boxes = get_bev_boxes(self)
            if self.cfg.SAVE_SENSORS:
                rgb_front = []
                rgb_front.append(cv2.cvtColor(input_data['rgb_front'][1][:, :, :3], cv2.COLOR_BGR2RGB)) ## TODO
                rgb_front = np.concatenate(rgb_front, axis=1)

                rgb_left = []
                rgb_left.append(cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB))
                rgb_left = np.concatenate(rgb_left, axis=1)

                rgb_right = []
                rgb_right.append(cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB))
                rgb_right = np.concatenate(rgb_right, axis=1)

                # rgb_augmented = input_data['rgb_augmented'][1][:, :, :3]
                # lidar = input_data['lidar']
            else:
                rgb_front = None
                rgb_left = None
                rgb_right = None

            
        else:
            boxes = None

        result.update({
            'rgb_front': rgb_front,
            'rgb_left': rgb_left,
            'rgb_right': rgb_right,
            'boxes': boxes,
        })

        if 'my_labels' in input_data:
            result['my_labels'] = input_data['my_labels']

        bev = render_BEV(self)
        bev = np.asarray(bev.squeeze().cpu()) # (15, 300, 300)
        
        topdown = encode_npy_to_pil(bev)
        topdown=np.moveaxis(topdown,0,2)
        hdmap=np.moveaxis(bev[:2],0,2) # 300,300,2
        
        result['topdown'] = topdown
        result['hdmap'] = hdmap
        return result
    
    def get_bev_boxes(self, input_data, pos):
        return get_bev_boxes(self, input_data=input_data, pos=pos)

    @torch.no_grad()
    def run_step(self, input_data, timestamp, sensors=None):
        # Must be called before run_step, so that the correct augmentation shift is
        # Saved
        # if self.datagen:
        #     self.augment_camera(sensors) # TODO
        control = super().run_step(input_data, timestamp)

        if self.step % self.save_freq == 0:
            if self.save_path is not None:
                tick_data = self.tick(input_data)
                self.save_sensors(tick_data)

            if SHUFFLE_WEATHER and self.step % self.save_freq == 0:
                self.shuffle_weather()
            
            # _, _, _, _ = self.scenario_logger.log_step(self.waypoint_route[:10])
            
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


    def save_sensors(self, tick_data):#dict_keys(['gps', 'speed', 'compass', 'rgb_front', 'rgb_left', 'rgb_right', 'boxes', 'my_labels'])
        frame = self.step // self.save_freq
        
        if self.cfg.SAVE_SENSORS:
            cv2.imwrite(str(self.save_path / 'rgb_front' / (f'{frame:04}.png')), tick_data['rgb_front'])
            cv2.imwrite(str(self.save_path / 'rgb_left' / (f'{frame:04}.png')), tick_data['rgb_left'])
            cv2.imwrite(str(self.save_path / 'rgb_right' / (f'{frame:04}.png')), tick_data['rgb_right'])
            cv2.imwrite(str(self.save_path / 'topdown' / (f'{frame:04}.png')), tick_data['topdown'])
            with open(str(self.save_path / 'hdmap' / ('%04d.pkl' % frame)),'wb') as f: 
                pickle.dump(tick_data['hdmap'],f,protocol=pickle.HIGHEST_PROTOCOL)
            # cv2.imwrite(str(self.save_path / 'rgb_augmented' / (f'{frame:04}.png')), tick_data['rgb_augmented'])
            # np.save(self.save_path / 'lidar' / ('%04d.npy' % frame), tick_data['lidar'], allow_pickle=True)
            inp_dict, gt_pts_bbox = get_batch_inp_ndetr(self, label_raw=tick_data['boxes'], input_data=tick_data)
        
            time_and_vis(self, inp_dict, gt_pts_bbox, pred_pts_bbox={})

        self.save_labels(self.save_path / 'boxes' / ('%04d.json' % frame), tick_data['boxes'])
        
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