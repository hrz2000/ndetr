import hydra
from torch.utils.data import random_split
from projects.mmdet3d_plugin.models.detectors.detr3d import get_k
from tools.preprocess.carla_dataset import PlanTDataset as Dataset
from tqdm import tqdm
import numpy as np
import pickle
import mmcv
from projects.configs.detr3d.new.common import class_names

@hydra.main(config_path=f"./config", config_name="config", version_base=None)
def main(cfg):
    # print(OmegaConf.to_yaml(cfg))
    shared_dict = None
    dataset_type = 'all' # all, 2all, sub
    
    if cfg.benchmark == 'lav':
        # we train without T2 and T5
        print(f'LAV training without T2 and T5')
        train_set = Dataset(cfg.data_dir, cfg, shared_dict=shared_dict, split='train')
        val_set = Dataset(cfg.data_dir, cfg, shared_dict=shared_dict, split='val')
    
    elif cfg.benchmark == 'longest6':
        print(f'Longest6 training with all towns')
        # train_set = Dataset(
        #     cfg.data_dir, cfg, shared_dict=shared_dict, split="train"
        # )
        # val_set = Dataset(
        #     cfg.data_dir, cfg, shared_dict=shared_dict, split="val"
        # )
        # dataset = Dataset(
        #     cfg.data_dir, cfg, shared_dict=shared_dict, split="all"
        # )
        # train_length = int(len(dataset) * 0.98)
        # val_length = len(dataset) - train_length
        # train_set, val_set = random_split(dataset, [train_length, val_length])
        # train_set = dataset
        # val_set = dataset
        
        if dataset_type=='all':
            val_set = Dataset('output/datagen_l6', cfg, shared_dict=shared_dict, split="all")
            train_set = Dataset('output/plant_datagen/PlanT_data_1', cfg, shared_dict=shared_dict, split="all")
        elif dataset_type=='2all':
            val_set = Dataset('output/datagen_l6', cfg, shared_dict=shared_dict, split="all")
            train_set = Dataset('output/plant_datagen/PlanT_data_1', cfg, shared_dict=shared_dict, split="all")
            train_set2 = Dataset('output/plant_datagen2/PlanT_data_1', cfg, shared_dict=shared_dict, split="all")
            train_set = train_set + train_set2
        else:
            train_set = Dataset(cfg.data_dir, cfg, shared_dict=shared_dict, split="train")
            val_set = Dataset(cfg.data_dir, cfg, shared_dict=shared_dict, split="val")
    else: 
        raise ValueError(f"Unknown benchmark: {cfg.benchmark}")
        
    print(f'Validation set size: {len(val_set)}')
    print(f'Train set size: {len(train_set)}')

    intri, vl2cam, cam2vl_r, cam2vl_t, cam2vl = get_k()
    
    for k,dataset in [('val',val_set), ('train',train_set)]:
        if dataset_type=='all':
            name = f'data/carla_{k}_hdmap_all_filter.pkl'
        elif dataset_type=='2all':
            name = f'data/carla_{k}_hdmap_2all_filter.pkl'
        else:
            name = f'data/carla_{k}_hdmap_filter.pkl'
        load_dataset(k, dataset, intri, cam2vl_r, cam2vl_t, name)

filter_all = mmcv.list_from_file('filter_all.log')

def load_dataset(k, dataset, intri, sensor2vl_r, sensor2vl_t, name):
    dataset[0]
    data_infos = []
    route_num = {}
    command_num = {}
    clas_num = {0:0,1:0}
    for idx, item in enumerate(tqdm(dataset)):
        if item==None:
            continue
        #dict_keys(['img_metas', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 'plan_metas'])
        token = f"{idx:05d}"
        gt_boxes = item['gt_bboxes_3d'].tensor.numpy() # (4, 9)
        gt_labels = item['gt_labels_3d'].astype(int)
        gt_idxs = np.array(item['gt_idxs'])
        clas_num[0] += sum(gt_labels==0)
        clas_num[1] += sum(gt_labels==1)
        classes = np.array([t.lower() for t in class_names])
        # array(['car', 'pedestrian'], dtype='<U10')
        gt_names = classes[gt_labels]
        
        fv = item['img_metas']['img_filename']
        if fv in filter_all:
            print('skip one')
            continue

        item = dict(
            lidar_path = None,
            token = token,
            sweeps = None,
            cams = dict(CAM_FRONT=dict(
                data_path = item['img_metas']['img_filename'],
                type = 'CAM_FRONT',
                sample_data_token = token,
                sensor2ego_translation = None,
                sensor2ego_rotation = None,
                ego2global_translation = None,
                ego2global_rotation = None,
                timestamp = 0,
                sensor2lidar_rotation = sensor2vl_r,
                sensor2lidar_translation = sensor2vl_t,
                cam_intrinsic = intri,
            )),
            lidar2ego_translation = item['img_metas']['lidar2ego_translation'],
            lidar2ego_rotation = item['img_metas']['lidar2ego_rotation'],
            ego2global_translation = None,
            ego2global_rotation = None,
            timestamp = 0,
            gt_boxes = gt_boxes,
            gt_names = gt_names,
            gt_idxs = gt_idxs,
            gt_bev = item['img_metas']['topdown_path'],
            plan = item['img_metas']['plan'],
            num_lidar_pts = -1, # None会报错
            num_radar_pts = None,
            valid_flag = np.ones((len(gt_boxes))).astype(bool),
        )
        num = len(item['plan']['route'])
        if num not in route_num:
            route_num[num] = 0
        route_num[num] += 1

        command = item['plan']['command']
        if command not in command_num:
            command_num[command] = 0
        command_num[command] += 1
        
        data_infos.append(item)
    print("route_num",route_num)
    print("command_num",command_num)
    print("clas_num",clas_num)
    print(len(data_infos))
    output = dict(
        infos = data_infos,
        metadata = {'version':f'carla_{k}'}
    )
    with open(name,'wb') as f: 
        pickle.dump(output,f,protocol=pickle.HIGHEST_PROTOCOL)

main()