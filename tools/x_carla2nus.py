import hydra
from torch.utils.data import random_split
from projects.mmdet3d_plugin.models.detectors.detr3d import get_k
from training.PlanT.dataset import PlanTDataset as Dataset
from tqdm import tqdm
import numpy as np
import pickle
from projects.configs.detr3d.new.common import class_names

@hydra.main(config_path=f"./config", config_name="config", version_base=None)
def main(cfg):
    # print(OmegaConf.to_yaml(cfg))
    shared_dict = None
    dataset_type = '2all' # all, 2all, sub
    
    if cfg.benchmark == 'lav':
        # we train without T2 and T5
        print(f'LAV training without T2 and T5')
        train_set = Dataset(cfg.data_dir, cfg, shared_dict=shared_dict, split='train')
        val_set = Dataset(cfg.data_dir, cfg, shared_dict=shared_dict, split='val')
    
    elif cfg.benchmark == 'longest6':
        print(f'Longest6 training with all towns')
        
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
            name = f'data/carla_{k}_hdmap_all.pkl'
        elif dataset_type=='2all':
            name = f'data/carla_{k}_hdmap_2all.pkl'
        else:
            name = f'data/carla_{k}_hdmap.pkl'
        load_dataset(k, dataset, intri, cam2vl_r, cam2vl_t, name)

def load_dataset(k, dataset, intri, sensor2vl_r, sensor2vl_t, name):
    dataset[0]
    data_infos = []
    route_dict = {}
    command_dict = {}
    clas_dict = {0:0,1:0}
    lidar2ego_translation=np.array([ 1.3, 0. ,  2.5]),
    lidar2ego_rotation=np.eye(3),
    for idx, item in enumerate(tqdm(dataset)): # 这里不进行dataloader操作，不会fn
        # dict_keys(['input', 'output', 'brake', 'waypoints', 'target_point', 'light', 'input_idx', 'input_label_path', 'output_idx', 'output_disappear', 'output_label_path'])
        fv_path = item['input_label_path']
        gt_boxes, gt_labels, gt_idxs = extract_data(item)
        plan = dict(
            wp=item['waypoints'],
            tp=item['target_point'],
            light=item['light'],
            route=route,
            command=command
        )
        
        classes = np.array([t.lower() for t in class_names]) 
        gt_names = classes[gt_labels] # array(['car', 'pedestrian'], dtype='<U10')

        item = dict(
            lidar_path = None,
            token = None, # dataset里面设置为fv_path
            sweeps = None,
            cams = dict(CAM_FRONT=dict(
                data_path = fv_path,
                type = 'CAM_FRONT',
                sample_data_token = None,
                sensor2ego_translation = None,
                sensor2ego_rotation = None,
                ego2global_translation = None,
                ego2global_rotation = None,
                timestamp = 0,
                sensor2lidar_rotation = sensor2vl_r,
                sensor2lidar_translation = sensor2vl_t,
                cam_intrinsic = intri,
            )),
            lidar2ego_translation = lidar2ego_translation,
            lidar2ego_rotation = lidar2ego_rotation,
            ego2global_translation = None,
            ego2global_rotation = None,
            timestamp = 0,
            gt_boxes = gt_boxes,
            gt_names = gt_names,
            gt_idxs = gt_idxs,
            plan = plan,
            num_lidar_pts = -1, # None会报错
            num_radar_pts = None,
            valid_flag = np.ones((len(gt_boxes))).astype(bool),
        )

        clas_dict[0] += sum(gt_labels==0)
        clas_dict[1] += sum(gt_labels==1)

        num = len(plan['route'])
        if num not in route_dict:
            route_dict[num] = 0
        route_dict[num] += 1

        command = plan['command']
        if command not in command_dict:
            command_dict[command] = 0
        command_dict[command] += 1
        
        data_infos.append(item)
        
    print("route_dict",route_dict)
    print("command_dict",command_dict)
    print("clas_dict",clas_dict)
    
    print(len(data_infos))
    
    output = dict(
        infos = data_infos,
        metadata = {'version':f'carla_{k}'}
    )
    with open(name,'wb') as f: 
        pickle.dump(output,f,protocol=pickle.HIGHEST_PROTOCOL)

def extract_data(item):
    gt_boxes = item['gt_bboxes_3d'].tensor.numpy() # (4, 9)
    gt_labels = item['gt_labels_3d'].astype(int)
    gt_idxs = item['input_idx']

main()