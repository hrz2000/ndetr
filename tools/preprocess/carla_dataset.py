import os
import sys
import glob
import logging
import json
import copy
import math
import numpy as np
from pathlib import Path
from einops import rearrange
import math

import torch
from torch.utils.data import Dataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from projects.mmdet3d_plugin.models.detectors.detr3d import get_k
from projects.configs.detr3d.new.common import class_names
from projects.mmdet3d_plugin.datasets.nuscenes_dataset import rel2s3

intri, vl2cam, cam2vl_r, cam2vl_t, cam2vl = get_k()
cam2img = intri @ vl2cam

class PlanTDataset(Dataset):
    # @beartype
    def __init__(self, root: str, cfg, shared_dict=None, split: str = "all") -> None:
        self.cfg = cfg
        self.cfg_train = cfg.model.training
        self.data_cache = shared_dict
        self.cnt = 0

        self.input_sequence_files = []
        self.output_sequence_files = []
        self.labels = []
        self.measurements = []

        with open('./data/wrong.txt','w') as f:
            pass
    
        label_raw_path_all = glob.glob(root + "/**/Routes*", recursive=True)
        # 'output/datagen_l6/**/Routes*'
        # 'output/plant_datagen/PlanT_data_1/**/Routes*'
        label_raw_path = []

        label_raw_path = self.filter_data_by_town(label_raw_path_all, split)
            
        print(f"Found {len(label_raw_path)} Route folders")

        # add multiple datasets (different seeds while collecting data)
        if cfg.trainset_size >= 2:
            add_data_path = root[:-2] + "_2"
            label_add_path_all = glob.glob(
                add_data_path + "/**/Routes*", recursive=True
            )
            label_add_path = self.filter_data_by_town(label_add_path_all, split)
            label_raw_path += label_add_path
        if cfg.trainset_size >= 3:
            add_data_path = root[:-2] + "_3"
            label_add_path_all = glob.glob(
                add_data_path + "/**/Routes*", recursive=True
            )
            label_add_path = self.filter_data_by_town(label_add_path_all, split)
            label_raw_path += label_add_path
        if cfg.trainset_size >= 4:
            raise NotImplementedError

        logging.info(f"Found {len(label_raw_path)} Route folders containing {cfg.trainset_size} datasets.")

        for sub_route in label_raw_path:
            root_files = os.listdir(sub_route) # ['topdown', 'lidar', 'hdmap', 'boxes', 'measurements', 'rgb'] 是不对的
            routes = [
                folder
                for folder in root_files
                if not os.path.isfile(os.path.join(sub_route, folder))
            ]
            # import pdb;pdb.set_trace()
            for route in routes:
                route_dir = Path(f"{sub_route}/{route}")
                wrong = False
                for k in ['rgb','boxes','measurements']: # check
                    if not os.path.exists(route_dir / k):
                        wrong = True
                        break
                if wrong:
                    with open('./data/wrong.txt','a') as f:
                        f.write(str(route_dir)+' loss_dir\n')
                    continue
                num_seq = len(os.listdir(route_dir / "boxes"))
                if not num_seq: # check
                    with open('./data/wrong.txt','a') as f:
                        f.write(str(route_dir)+' num=0\n')
                    continue

                # ignore the first 5 and last two frames
                for seq in range(
                    5,
                    num_seq - self.cfg_train.pred_len - self.cfg_train.seq_len - 2,
                ):
                    # load input seq and pred seq jointly
                    label = []
                    measurement = []
                    for idx in range(
                        self.cfg_train.seq_len + self.cfg_train.pred_len
                    ):
                        labels_file = route_dir / "boxes" / f"{seq + idx:04d}.json"
                        measurements_file = (
                            route_dir / "measurements" / f"{seq + idx:04d}.json"
                        )
                        label.append(labels_file)
                        measurement.append(measurements_file)
                    self.labels.append(label)
                    self.measurements.append(measurement)
                    
        

        # There is a complex "memory leak"/performance issue when using Python objects like lists in a Dataloader that is loaded with multiprocessing, num_workers > 0
        # A summary of that ongoing discussion can be found here https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # A workaround is to store the string lists as numpy byte objects because they only have 1 refcount.
        self.labels       = np.array(self.labels      ).astype(np.string_)
        self.measurements = np.array(self.measurements).astype(np.string_)
        # self.labels       = self.labels[:100]
        # self.measurements = self.measurements[:100]
        print(f"Loading {len(self.labels)} samples from {root}")


    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.measurements)


    def __getitem__(self, index):
        """Returns the item at index idx."""

        labels = self.labels[index]
        measurements = self.measurements[index]
    
        sample = {
            "input": [],
            "output": [],
            "brake": [],
            "waypoints": [],
            "target_point": [],
            "light": [],
        }
        if not self.data_cache is None and labels[0] in self.data_cache:
            sample = self.data_cache[labels[0]]
        else:
            loaded_labels = []
            loaded_measurements = []

            for i in range(self.cfg_train.seq_len + self.cfg_train.pred_len):
                measurements_i = json.load(open(measurements[i]))
                labels_i = json.load(open(labels[i]))

                loaded_labels.append(labels_i)
                loaded_measurements.append(measurements_i)

            # ego car is always the first one in label file
            waypoints = get_waypoints(loaded_measurements[self.cfg_train.seq_len - 1 :])
            waypoints = transform_waypoints(waypoints)

            # save waypoints in meters
            filtered_waypoints = []
            for id in ["1"]:
                waypoint = []
                for matrix, _ in waypoints[id][1:]:
                    waypoint.append(matrix[:2, 3])
                filtered_waypoints.append(waypoint)
            waypoints = np.array(filtered_waypoints)

            ego_waypoint = waypoints[-1]

            sample["waypoints"] = ego_waypoint
      
            local_command_point = np.array(loaded_measurements[self.cfg_train.seq_len - 1]["target_point"])
            # local_command_point = np.array([loaded_measurements[self.cfg_train.seq_len - 1]["x_command"], loaded_measurements[self.cfg_train.seq_len - 1]["x"]]) - np.array([loaded_measurements[self.cfg_train.seq_len - 1]["y_command"],loaded_measurements[self.cfg_train.seq_len - 1]["y"]])
            sample["target_point"] = tuple(local_command_point)
            sample["light"] = loaded_measurements[self.cfg_train.seq_len - 1][
                "light_hazard"
            ]

            if self.cfg.model.pre_training.pretraining == "forecast":
                offset = (
                    self.cfg.model.pre_training.future_timestep
                )  # target is next timestep
            elif self.cfg.model.pre_training.pretraining == "none":
                offset = 0
            else:
                print(
                    f"ERROR: pretraining {self.cfg.model.pre_training.pretraining} is not supported"
                )
                sys.exit()

            for sample_key, file in zip(
                ["input", "output"],
                [
                    (
                        loaded_measurements[self.cfg_train.seq_len - 1],
                        loaded_labels[self.cfg_train.seq_len - 1],
                    ),
                    (
                        loaded_measurements[self.cfg_train.seq_len - 1 + offset],
                        loaded_labels[self.cfg_train.seq_len - 1 + offset],
                    ),
                ],
            ):
                if sample_key == 'output':
                    continue
                    
                # labels_data_all = file[1]
                labels_data = file[1]
                data_car, data_route, obj_idxs = extract_data(labels_data, self.cfg_train.max_NextRouteBBs)

                sample[sample_key] = data_car + data_route

        self.cnt+=1

        inp = np.array(sample['input']) # list:8, 每个元素包含7项->9项，route和car一样

        instance_t, gt_labels_3d, data_route = extract_data2(inp)

        wp = sample['waypoints'] # lidar系下
        tp = np.array(sample['target_point']) # ego系下
        light = sample['light']
        command = loaded_measurements[0]['command'] # 注意上面循环需要是input
        gt_idxs = obj_idxs
        

        ret = dict(
            img_metas = dict(
                img_filename=labels[0].astype('str').replace('boxes','rgb').replace('.json','.png'),
                topdown_path=labels[0].astype('str').replace('boxes','topdown').replace('.json','.png'),
                plan = dict(
                    wp=wp,
                    tp=tp,
                    light=light,
                    route=data_route,
                    command=command,
                ),
                lidar2ego_translation=np.array([ 1.3, 0. ,  2.5]),
                lidar2ego_rotation=np.eye(3),
            ),
            gt_bboxes_3d = LiDARInstance3DBoxes(instance_t,box_dim=instance_t.shape[-1]),
            gt_labels_3d = gt_labels_3d,
            gt_idxs = gt_idxs,
        )
        return ret

    def quantize_box(self, boxes):
        boxes = np.array(boxes)

        # range of xy is [-30, 30]
        # range of yaw is [-360, 0]
        # range of speed is [0, 60]
        # range of extent is [0, 30]

        # quantize xy
        boxes[:, 1] = (boxes[:, 1] + 30) / 60
        boxes[:, 2] = (boxes[:, 2] + 30) / 60

        # quantize yaw
        boxes[:, 3] = (boxes[:, 3] + 360) / 360

        # quantize speed
        boxes[:, 4] = boxes[:, 4] / 60

        # quantize extent
        boxes[:, 5] = boxes[:, 5] / 30
        boxes[:, 6] = boxes[:, 6] / 30

        boxes[:, 1:] = np.clip(boxes[:, 1:], 0, 1)

        size_pos = pow(2, self.cfg.model.pre_training.precision_pos)
        size_speed = pow(2, self.cfg.model.pre_training.precision_speed)
        size_angle = pow(2, self.cfg.model.pre_training.precision_angle)

        boxes[:, [1, 2, 5, 6]] = (boxes[:, [1, 2, 5, 6]] * (size_pos - 1)).round()
        boxes[:, 3] = (boxes[:, 3] * (size_angle - 1)).round()
        boxes[:, 4] = (boxes[:, 4] * (size_speed - 1)).round()

        return boxes.astype(np.int32).tolist()


    def filter_data_by_town(self, label_raw_path_all, split):
        # in case we want to train without T2 and T5
        label_raw_path = []
        if split == "train":
            for path in label_raw_path_all:
                if "Town02" in path:# or "Town05" in path:
                    continue
                label_raw_path.append(path)
        elif split == "val":
            for path in label_raw_path_all:
                if "Town02" in path:# or "Town05" in path:
                    label_raw_path.append(path)
        elif split == "all":
            label_raw_path = label_raw_path_all
            
        return label_raw_path

def split_large_BB(route, start_id):
    x = route[1]
    y = route[2]
    angle = route[3] - 90
    extent_x = route[5] / 2
    extent_y = route[6] / 2

    x1 = x - extent_y * math.sin(math.radians(angle))
    y1 = y - extent_y * math.cos(math.radians(angle))

    x0 = x + extent_y * math.sin(math.radians(angle))
    y0 = y + extent_y * math.cos(math.radians(angle))

    number_of_points = (
        math.ceil(extent_y * 2 / 10) - 1
    )  # 5 is the minimum distance between two points, we want to have math.ceil(extent_y / 5) and that minus 1 points
    xs = np.linspace(
        x0, x1, number_of_points + 2
    )  # +2 because we want to have the first and last point
    ys = np.linspace(y0, y1, number_of_points + 2)

    splitted_routes = []
    for i in range(len(xs) - 1):
        route_new = route.copy()
        route_new[1] = (xs[i] + xs[i + 1]) / 2
        route_new[2] = (ys[i] + ys[i + 1]) / 2
        route_new[4] = float(start_id + i)
        route_new[5] = extent_x * 2
        route_new[6] = route[6] / (
            number_of_points + 1
        )
        splitted_routes.append(route_new)

    return splitted_routes


def get_waypoints(measurements):
    assert len(measurements) == 5
    num = 5
    waypoints = {"1": []}

    for i in range(0, num):
        waypoints["1"].append([measurements[i]["ego_matrix"], True])

    Identity = list(list(row) for row in np.eye(4))
    # padding here
    for k in waypoints.keys():
        while len(waypoints[k]) < num:
            waypoints[k].append([Identity, False])
    return waypoints


def transform_waypoints(waypoints):
    """transform waypoints to be origin at ego_matrix"""

    # TODO should transform to virtual lidar coordicate?
    T = get_vehicle_to_virtual_lidar_transform()

    for k in waypoints.keys():
        vehicle_matrix = np.array(waypoints[k][0][0])
        vehicle_matrix_inv = np.linalg.inv(vehicle_matrix)
        for i in range(1, len(waypoints[k])):
            matrix = np.array(waypoints[k][i][0])
            waypoints[k][i][0] = T @ vehicle_matrix_inv @ matrix

    return waypoints


def get_virtual_lidar_to_vehicle_transform():
    # This is a fake lidar coordinate
    T = np.eye(4)
    T[0, 3] = 1.3
    T[1, 3] = 0.0
    T[2, 3] = 2.5
    return T


def get_vehicle_to_virtual_lidar_transform():
    return np.linalg.inv(get_virtual_lidar_to_vehicle_transform())


def generate_batch(data_batch):
    input_batch, output_batch = [], []
    for element_id, sample in enumerate(data_batch):
        input_item = torch.tensor(sample["input"], dtype=torch.float32)
        output_item = torch.tensor(sample["output"])

        input_indices = torch.tensor([element_id] * len(input_item)).unsqueeze(1)
        output_indices = torch.tensor([element_id] * len(output_item)).unsqueeze(1)

        input_batch.append(torch.cat([input_indices, input_item], dim=1))
        output_batch.append(torch.cat([output_indices, output_item], dim=1))

    waypoints_batch = torch.tensor([sample["waypoints"] for sample in data_batch])
    tp_batch = torch.tensor(
        [sample["target_point"] for sample in data_batch], dtype=torch.float32
    )
    light_batch = rearrange(
        torch.tensor([sample["light"] for sample in data_batch]), "b -> b 1"
    )

    return input_batch, output_batch, waypoints_batch, tp_batch, light_batch

def extract_data(labels_data, max_NextRouteBBs):
    # light, lane, route
    nm2id = {'Car':1.0, 'Pedestrian':1.1}
    data_car = [ # 我这个只包含第一帧的信息，第二帧的话就会出错
        [
            nm2id[x['class']],  # type indicator for cars
            float(x["position"][0]),
            float(x["position"][1]),
            float(x["yaw"] * 180 / 3.14159265359),  # in degrees
            float(x["speed"] * 3.6),  # in km/h
            float(x["extent"][2]),
            float(x["extent"][1]),

            float(x["position"][2]),
            float(x["extent"][0]),
        ]
        for j, x in enumerate(labels_data) if x["class"].lower() in class_names and float(x["position"][0]) >= -1.3
    ]
    
    obj_idxs = [float(x["id"]) for j, x in enumerate(labels_data) if x["class"].lower() in class_names and float(x["position"][0]) >= -1.3] # 这是gt里面，其中包含car和pede
    obj_idxs = np.array(obj_idxs)

    # if we use the far_node as target waypoint we need the route as input
    # 与car不同，route在ego系
    data_route = [
        [
            2.0,  # type indicator for route
            float(x["position"][0]) - float(labels_data[0]["position"][0]),
            float(x["position"][1]) - float(labels_data[0]["position"][1]),
            float(x["yaw"] * 180 / 3.14159265359),  # in degrees
            float(x["id"]), # 没有速度
            float(x["extent"][2]),
            float(x["extent"][1]),

            float(x["position"][2]) - float(labels_data[0]["position"][2]),
            float(x["extent"][0]),
        ]
        for j, x in enumerate(labels_data)
        if x["class"] == "Route"
        and float(x["id"]) < max_NextRouteBBs
    ]
    
    # we split route segment slonger than 10m into multiple segments
    # improves generalization
    data_route_split = []

    for route in data_route:
        if route[6] > 10: # 前后方向的长度太大
            routes = split_large_BB( # 增加的两个维度不影响
                route, len(data_route_split)
            )
            data_route_split.extend(routes)
        else:
            data_route_split.append(route)

    data_route = data_route_split[: max_NextRouteBBs]
    if len(data_route) == 0:
        print("ERROR: no route found")
        sys.exit()
    # data_route = np.array(data_route)[:,1:7]

    return data_car, data_route, obj_idxs

def extract_data2(inp):
    typ = inp[:,0]
    # pos0 = inp[:,1]
    # pos1 = inp[:,2]
    # yawdu = inp[:,3]
    # speed3_or_id = inp[:,4]
    # ext2 = inp[:,5]
    # ext1 = inp[:,6]
    # pos2 = inp[:,7]
    # ext0 = inp[:,8]
    instance_t = inp[:,[1,2,7,6,5,8,3,4,4]][np.logical_or(typ==1.0,typ==1.1)] # (5, 8)
    instance_t[:,-2:] /= 3.6
    instance_t[:,6] = -instance_t[:,6]/180*np.pi
    instance_t[:,-2] = instance_t[:,-2] * np.sin(instance_t[:,6])
    instance_t[:,-1] = instance_t[:,-1] * np.cos(instance_t[:,6])
    # (6, 9)
    id2label = {1.0:0, 1.1:1}
    gt_labels_3d = np.array([id2label[t] for t in typ if t==1.0 or t==1.1])
    assert len(instance_t) == len(gt_labels_3d)

    data_route = inp[:,1:7][typ==2]
    data_route[:,2] = -data_route[:,2]/180*np.pi # 
    return instance_t, gt_labels_3d, data_route