import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset,Custom3DDataset
import tempfile
from os import path as osp
import mmcv
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
import torch
import matplotlib.pyplot as plt

from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.datasets.pipelines import Compose
import torch
import numpy as np
import mmcv
from mmdet3d.core.bbox import limit_period, BaseInstance3DBoxes
from .vis_tools import create_front, create_bev, create_fut_bev, create_collide_bev
import copy
from mmcv.parallel import DataContainer as DC
from projects.configs.detr3d.new.cc import on_cc, enable_mc

@DATASETS.register_module()
class CustomNuScenesDataset(Custom3DDataset):

    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    AttrMapping = {
        'cycle.with_rider': 0,
        'cycle.without_rider': 1,
        'pedestrian.moving': 2,
        'pedestrian.standing': 3,
        'pedestrian.sitting_lying_down': 4,
        'vehicle.moving': 5,
        'vehicle.parked': 6,
        'vehicle.stopped': 7,
    }
    AttrMapping_rev = [
        'cycle.with_rider',
        'cycle.without_rider',
        'pedestrian.moving',
        'pedestrian.standing',
        'pedestrian.sitting_lying_down',
        'vehicle.moving',
        'vehicle.parked',
        'vehicle.stopped',
    ]
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False,
                 img_shape=None,
                 is_carla=False,
                 vis=False,
                 vis_dir=None,
                 clss_range=None,
                 debug=False,
                 use_det_metric=False,
                 in_test=False,
                 **kwargs):
        self.use_det_metric = use_det_metric 
        self.in_test = in_test
        self.prev = 3
        self.n_fut = 4
        # self.queue_length = self.prev+1+self.n_fut
        self.queue_length = self.prev + 1
        self.temporal = kwargs['temporal']
        self.debug = debug
        self.vis_dir = vis_dir
        self.vis = vis
        self.is_carla = is_carla
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)
        self.img_shape = img_shape

        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory(self.eval_version)

        if is_carla and clss_range!=None:
            self.eval_detection_configs.clss_range = clss_range
            self.eval_detection_configs.class_names = self.CLASSES

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        if self.debug: # TODO
            # import pdb;pdb.set_trace()
            # data_infos = data_infos[1185:1195]
            # data_infos = [data_infos[i] for i in [20,30,40,50,100,500,700,800,900,1200,1190]]
            data_infos = data_infos[1185:1300]
        elif self.in_test:
            # data_infos = data_infos[:5]
            data_infos = data_infos[0:1210]
        return data_infos

    def get_data_info(self, index):
        
        info = self.data_infos[index] # dict_keys(['lidar_path', 'token', 'sweeps', 'cams', 'lidar2ego_translation', 'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation', 'timestamp', 'gt_boxes', 'gt_names', 'gt_bev', 'plan', 'num_lidar_pts', 'num_radar_pts', 'valid_flag'])

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            for cam_type, cam_info in info['cams'].items():
                if not on_cc:
                    image_paths.append(cam_info['data_path'])
                else:
                    image_paths.append(rel2s3(cam_info['data_path']))
                    
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T # li2cam的平移矩阵的计算
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                break
            
        cam2imgs = lidar2img_rts

        plan=info['plan']
        fv_path = image_paths[0]
        topdown_path = fv_path.replace('rgb','topdown')
        if on_cc:
            topdown_path = rel2s3(topdown_path)
        topdown = mmcv.imread(topdown_path)

        hdmap = np.stack([
            mmcv.imread(fv_path.replace('rgb','hdmap0'),'grayscale'), 
            mmcv.imread(fv_path.replace('rgb','hdmap1'),'grayscale')], axis=0) # 其实保存的时候用pkl会更好一些
            
        attnmap_path = fv_path.replace('rgb', 'attnmap').replace('png','pkl')
        attn_info = mmcv.load(attnmap_path, file_format='pkl') # dict_keys(['attn_map', 'input_idx', 'output_idx', 'output_disappear', 'output_label_path'])
            
        ann_info = None
        if not self.test_mode:
            ann_info = self.get_ann_info(index, cam2imgs, attn_info) # 这里面进行了过滤

        input_dict = dict(
            # sample_idx=info['sample_idx'],
            sample_idx=fv_path,
            scenes=osp.dirname(image_paths[0]),
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
            img_filename=image_paths,
            lidar2img=lidar2img_rts, # TODO
            cam2imgs=cam2imgs, # 不是bs是cams
            ann_info=ann_info,
            plan=plan,
            topdown_path=topdown_path,
            topdown=topdown,
            hdmap=hdmap,
            attn_info=attn_info,
            gt_idxs=ann_info['gt_idxs'],
            attnmap=ann_info['attnmap'],
            gt_pl_lack=ann_info['gt_pl_lack'],
            index=index,
        )
        
        return input_dict

    def get_ann_info(self, index, cam2imgs, attn_info): # 只要保证经过这个东西之后，attn_weight和还是1就行
        # get_data_info里面没有对gt_idxs等进行过滤处理
        info = self.data_infos[index]

        ## attn
        attnmap = attn_info['attn_map'] # (8, 8, 42, 42)
        input_idx = attn_info['input_idx'] # 21
        output_idx = attn_info['output_idx'] # 理论上inp和oup的idx是完全一样的
        # 后面的0,1代表route的idx
        assert len(input_idx) == len(output_idx)
        output_disappear = attn_info['output_disappear']
        
        # TODO
        
        len_box_route = len(input_idx)
        len_box = len(output_disappear) # 21
        plant_gt_idxs = input_idx[:len_box] # bugfix：注意这里没有cls_emb
        len_route = len_box_route - len_box
        # 让预测跟着gt走就行
        # init_box_num = len(info['gt_boxes'])
        # boxes_id = list(range(init_box_num)) # TODO: 目前两个gt不一样，导致后面mask是不对的
        # attnmap和plant_gt_idxs是一致的
        # gt_idxs和其他是一致的
        
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask] # (66, 9)
        gt_names_3d = info['gt_names'][mask] #类别名字
        gt_idxs = info['gt_idxs'][mask] #类别名字
        # boxes_id = np.array(boxes_id)[mask] # 必须是np才能用yes/no进行筛选
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)
        
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1])
        ego_box_3d = gt_bboxes_3d[0]

        cam2img = cam2imgs[0]
        valid_mask = filter_invisible(gt_bboxes_3d, self.img_shape, cam2img) # 对gt也进行了这样的处理，至少没有了ego
        gt_bboxes_3d = gt_bboxes_3d[valid_mask]
        gt_labels_3d = gt_labels_3d[valid_mask]
        gt_names_3d = gt_names_3d[valid_mask]
        gt_idxs = gt_idxs[valid_mask]
        # boxes_id = boxes_id[valid_mask]
        
        # 这个时候找gt_idxs和plant_gt_idxs的共同之处
        pl_box_idxs_raw = []
        for idx in gt_idxs:
            # import pdb;pdb.set_trace()
            idx_pl = np.where(plant_gt_idxs==idx)[0] # 获得了5
            if idx_pl.shape[0] == 0:
                pl_box_idxs_raw.append(-1)
            else:
                idx_pl = idx_pl.item()
                pl_box_idxs_raw.append(idx_pl) # 前面有wp_emb
                # pl_box_idxs_raw.append(idx_pl+1) # 前面有wp_emb
        
        # import pdb;pdb.set_trace()
        # pl_exists_idxs = np.array(pl_new_idxs)[pl_new_idxs!=-1][0] # pl有我没有的先不管
        pl_box_idxs = np.array(pl_box_idxs_raw)
        
        gt_pl_lack = pl_box_idxs==-1
        
        # if gt_pl_lack.sum() > 0:
        #     import pdb;pdb.set_trace()
        
        # import pdb;pdb.set_trace()
        # gt_idxs: [1501, 1510, 1549, 1600, 1603, 1674, 1706, 1716, 1732, 1736]
        #           [1,     3,   6,    9, 10, 16, 18, 19, 20, 21]
        
        # attnmap: array([0.00086509, 0.00366808, 0.02288333, 0.03979345, 0.04145063, 0.24237308, 0.04866292, 0.01946599, 0.03253264, 0.02445563,
        
        if len_box_route==1:
            pl_map_idxs = [0, *(pl_box_idxs.tolist()), len_box_route]
            gt_pl_lack = [False, *gt_pl_lack, False]
        else:
            pl_map_idxs = [0, *(pl_box_idxs.tolist()), len_box_route-1] # len_box_route, 目前只用第一个route
            gt_pl_lack = [False, *gt_pl_lack, False]
        
        attnmap = attnmap[:,:,pl_map_idxs][:,:,:,pl_map_idxs] # (8, 8, 13, 13)
        # 现在attnmap是1+box+route+1
        # gt_idxs于此无关，我们希望是一样的
        # 如果想修改成和gt一样的格式，需要把-1的位置给填充成别的东西
        # import pdb;pdb.set_trace()
        # 第二个维度是gt_idxs，后者是yes or no，其实后者就行
        attnmap[:,:,gt_pl_lack][:,:,:,gt_pl_lack] = -1 # 完全按照gt_idx顺序，前面多一个cls_emb，后面多route
        
        # assert attnmap.shape[-1] == 1 + len(gt_idxs) + len_route
        assert attnmap.shape[-1] == 1 + len(gt_idxs) + 1
        
        # import pdb;pdb.set_trace()
        wp_attn = attnmap[-1,:,0,:] # wp对所有（wp、box、route）第一个head
        # wp_attn = attnmap[-1,:,0,:].mean(0) # wp对所有（wp、box、route）
        # 我们需要把wp_attn补全到和gt_idxs一样的格式
        # 对gt可视化的时候，前面concat上了ego
        # 第一个box其实是在idx=2的位置开始
    
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_idxs=gt_idxs,
            wp_attn=wp_attn,
            attnmap=attnmap,
            gt_pl_lack=gt_pl_lack,
            ego_box_3d=ego_box_3d
        )
        return anns_results

    def get_gt_pts_bbox(self, index): # 为了便于可视化的gt信息
        data_info = self.get_data_info(index)
        anns_results = data_info['ann_info']
        gt_bboxes_3d = anns_results['gt_bboxes_3d']
        ego_box_3d = anns_results['ego_box_3d']
        gt_labels_3d = anns_results['gt_labels_3d'] # 这里是没有过滤的
        gt_names = anns_results['gt_names']
        gt_idxs = anns_results['gt_idxs']
        wp_attn = anns_results['wp_attn']
        
        cam2img = data_info['cam2imgs'][0]
        imgpath = data_info['img_filename'][0] # TODO
            
        gt_pts_bbox = dict(
            boxes_3d=LiDARInstance3DBoxes.cat([ego_box_3d, gt_bboxes_3d]),
            scores_3d=np.ones_like([1,*gt_labels_3d]),
            labels_3d=np.array([1,*gt_labels_3d]),
            gt_idxs=np.array([0,*gt_idxs]), # 第一个是ego
            wp_attn=np.array([*wp_attn]), # 第一个是ego那个cls_emb，后面都正常索引
            
            attrs_3d=data_info['plan']['wp'],
            tp=data_info['plan']['tp'],
            light=data_info['plan']['light'],
            command=data_info['plan']['command'],
            route=data_info['plan']['route'],
            route_wp=None,
            iscollide=None,
            cam2img=cam2img,
            imgpath=imgpath,
            # gt_bev=data_info['topdown_path']
            # gt_bev=data_info['topdown_path']
            topdown=data_info['topdown'],#可视化的时候要用
            hdmap=None
        )
        return gt_pts_bbox

    def _format_bbox(self, results, jsonfile_prefix=""):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES#['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'] 10个

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            # det是一个dict里面包含3个元素，300个query
            annos = []
            boxes = output_to_nusc_box(det, self.with_velocity)#300个Box元素的list
            sample_token = self.data_infos[sample_id]['token']
            # boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
            #                                  mapped_class_names,
            #                                  self.eval_detection_configs,
            #                                  self.eval_version)#283个Box元素的list,这里面只是去除掉了一些太远的object,其实后面还会有filter
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]#'pedestrian'
                # box.velocity:array([-0.69207749,  0.88156534,  0.06821988])
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]#'pedestrian.moving'
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),#array([ 6.23096143e+02,  1.62322800e+03, -1.80724410e-02])
                    size=box.wlh.tolist(),#array([0.6692922 , 0.68082136, 1.7620763 ], dtype=float32)
                    rotation=box.orientation.elements.tolist(),#array([-0.68890765,  0.00624937, -0.03112052,  0.72415379])
                    velocity=box.velocity[:2].tolist(),#array([-0.69207749,  0.88156534])
                    detection_name=name,
                    detection_score=box.score,#0.851989209651947
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(self,
                         result_path,
                         gt_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        # from nuscenes.eval.detection.evaluate import NuScenesEval
        from .eval_tools import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])#'/tmp/tmpdn8kcyhe/results/pts_bbox'
        # if 'carla' not in self.version:
        #     nusc = NuScenes(
        #         version=self.version, dataroot=self.data_root, verbose=False)
        # else:
        nusc = None
        # eval_set_map = {
        #     'v1.0-mini': 'mini_val',
        #     'v1.0-trainval': 'val',
        #     'carla': 'carla'
        # }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            gt_path=gt_path,
            eval_set="",
            output_dir=output_dir,
            verbose=False)
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449

        # results[0].keys():dict_keys(['boxes_3d', 'scores_3d', 'labels_3d', 'attrs_3d'])
        # 包装之后变成 dict_keys(['pts_bbox'])
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:#dict_keys(['pts_bbox'])有81个
                if name not in ['pts_bbox']:
                    continue
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                #每个元素为dict_keys(['boxes_3d', 'scores_3d', 'labels_3d']) torch.Size([300])
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir
    
    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode: # 我都设置成了false
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue # 所以说训练和测试数量是一样的，但是可能对应不起来
            return data
        
    def evaluate(self,
                 batch_results, # results里面包含更多信息，才能对应起来
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 **kwargs):
        # len(results):323
        # results[0].keys():dict_keys(['pts_bbox'])
        # results[0]['pts_bbox'].keys():dict_keys(['boxes_3d', 'scores_3d', 'labels_3d'])
        # results[0]['pts_bbox']['boxes_3d'].tensor.shape:torch.Size([300,9])
        # results[0]['pts_bbox']['labels_3d'].shape:torch.Size([300])
        # results[0]里面有 pts_bbox 还有 loss，只处理前者
        # 过滤之后再进行metric的计算，在专门的decode中进行
        
        # result_files.keys():dict_keys(['pts_bbox'])
        # result_files['pts_bbox']:'/tmp/tmpk2skos1s/results/pts_bbox/results_nusc.json'
        # tmp_dir:<TemporaryDirectory '/tmp/tmpk2skos1s'>
        gt_results = []
        for i, result in enumerate(batch_results):
            pts_bbox = result['pts_bbox']
            end_idx = pts_bbox['img_metas']['index']
            gt_results.append(dict(pts_bbox=self.get_gt_pts_bbox(end_idx)))
        
        if self.use_det_metric and 'boxes_3d' in batch_results[0]['pts_bbox']:
            result_files, tmp_dir = self.format_results(batch_results, jsonfile_prefix)
            gt_result_files, tmp_dir2 = self.format_results(gt_results, jsonfile_prefix)
            
            if isinstance(result_files, dict):
                results_dict = dict()
                for name in result_names:
                    print('Evaluating bboxes of {}'.format(name))
                    ret_dict = self._evaluate_single(result_files[name], gt_result_files[name])
                    #'/tmp/tmpdn8kcyhe/results/pts_bbox/results_nusc.json'
                    results_dict.update(ret_dict)
            elif isinstance(result_files, str):
                results_dict = self._evaluate_single(result_files, gt_result_files)
            else:
                assert False

            if tmp_dir is not None:
                tmp_dir.cleanup()
            if tmp_dir2 is not None:
                tmp_dir2.cleanup()
        else:
            results_dict = {}

        if self.vis:
            self.show_ndetr(batch_results, gt_results, self.vis_dir, show=self.vis, pipeline=pipeline)
        
        # 对不同batch的loss进行平均
        loss_dict = {}
        for idx_r, result in enumerate(batch_results):
            result = result['loss']
            for k in result:
                if idx_r == 0:
                    loss_dict[k] = 0
                loss_dict[k] += result[k]
                if idx_r == len(batch_results) - 1:
                    loss_dict[k] /= len(batch_results)
        results_dict.update(loss_dict)

        return results_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show_ndetr(self, results, gt_results, out_dir, show=False, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        pipeline = self._get_pipeline(pipeline)
        for idx, (pred_pts_bbox, gt_pts_bbox) in enumerate(zip(results, gt_results)):
            if 'pts_bbox' in pred_pts_bbox.keys():
                pred_pts_bbox = pred_pts_bbox['pts_bbox']
                gt_pts_bbox = gt_pts_bbox['pts_bbox']
            show_results(idx, pred_pts_bbox, gt_pts_bbox, out_dir)
        
    def prepare_one_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example

    def prepare_train_data(self, index):
        if self.temporal == None:
            return self.prepare_one_train_data(index)
        elif self.temporal == 'bevformer':
            if index - self.queue_length + 1 < 0:
                return None
            queue = []
            index_list = list(range(index - self.queue_length + 1, index + 1))
            # 所以说index索引得到的最后当前帧，也就是这个index，但是index=0,1..可能对应None，在可视化的时候可能有问题
            for i in index_list:
                example = self.prepare_one_train_data(i)
                if example is None:
                    return None
                queue.append(example)
            if queue[0]['img_metas'].data['scenes'] != queue[-1]['img_metas'].data['scenes']:
                return None
            return self.union2one(queue)
        else:
            import pdb;pdb.set_trace()

    def union2one(self, queue):
        # TODO
        imgs_list = [each['img'].data for each in queue] # [torch.Size([1, 3, 256, 928]),...]
        metas_map = []
        # fut_gt_bboxes_3d_list = []
        # fut_gt_labels_3d_list = []
        # prev_scene_token = None
        # prev_pos = None
        # prev_angle = None
        for i, each in enumerate(queue): # dict_keys(['img_metas', 'gt_bboxes_3d', 'gt_labels_3d', 'img'])
            if i < self.prev+1: # 也就是所有帧
                img_metas = each['img_metas'].data
                img_metas['gt_bboxes_3d'] = each['gt_bboxes_3d'].data
                img_metas['gt_labels_3d'] = each['gt_labels_3d'].data
                metas_map.append(img_metas)
            # if i >= self.prev+1:
            #     fut_gt_bboxes_3d_list.append(each['gt_bboxes_3d']) # 理论上并不需要gt
            #     fut_gt_labels_3d_list.append(each['gt_labels_3d'])
        #     if metas_map[i]['scene_token'] != prev_scene_token:
        #         metas_map[i]['prev_bev_exists'] = False
        #         prev_scene_token = metas_map[i]['scene_token']
        #         prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
        #         prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
        #         metas_map[i]['can_bus'][:3] = 0
        #         metas_map[i]['can_bus'][-1] = 0
        #     else:
        #         metas_map[i]['prev_bev_exists'] = True
        #         tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
        #         tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
        #         metas_map[i]['can_bus'][:3] -= prev_pos
        #         metas_map[i]['can_bus'][-1] -= prev_angle
        #         prev_pos = copy.deepcopy(tmp_pos)
        #         prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]['img'] = DC(torch.stack(imgs_list[:self.prev+1]), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        # queue[-1]['fut_gt_bboxes_3d_list'] = DC(fut_gt_bboxes_3d_list, cpu_only=True)
        # queue[-1]['fut_gt_labels_3d_list'] = DC(fut_gt_labels_3d_list, cpu_only=True)
        queue = queue[-1]
        return queue

    def prepare_test_data(self, index):
        return self.prepare_train_data(index)

    def prepare_train_data_one(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)#dict_keys(['sample_idx', 'pts_filename', 'sweeps', 'timestamp', 'img_filename', 'lidar2img', 'cam_intrinsic', 'lidar2cam', 'ann_info'])
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)#增加了dict_keys(['img_fields', 'bbox3d_fields', 'pts_mask_fields', 'pts_seg_fields', 'bbox_fields', 'mask_fields', 'seg_fields', 'box_type_3d', 'box_mode_3d'])
        # import pdb;pdb.set_trace()
        example = self.pipeline(input_dict)#dict_keys(['img_metas', 'gt_bboxes_3d', 'gt_labels_3d', 'img'])其中img_metas是dict_keys(['filename', 'ori_shape', 'img_shape', 'lidar2img', 'pad_shape', 'scale_factor', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_idx', 'pts_filename'])
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None# t
        return example

    def prepare_test_data_one(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

from scipy.spatial.transform import Rotation
def make_mat_cam(data_info, mat):
    Rq = data_info['cams']['CAM_FRONT'][f'{mat}_rotation']
    Rm = Rotation.from_quat(Rq)
    rotation_matrix = Rm.as_matrix()
    cam2ego = np.eye(4)
    cam2ego[:3,:3] = rotation_matrix
    cam2ego[:3,-1] = data_info['cams']['CAM_FRONT'][f'{mat}_translation']
    return cam2ego

def make_mat(data_info, mat, is_carla):
    if not is_carla:
        Rq = data_info[f'{mat}_rotation']
        Rm = Rotation.from_quat(Rq)
        rotation_matrix = Rm.as_matrix()
    else:
        rotation_matrix = data_info[f'{mat}_rotation']
    cam2ego = np.eye(4)
    cam2ego[:3,:3] = rotation_matrix
    cam2ego[:3,-1] = data_info[f'{mat}_translation']
    return cam2ego

def show_results(index, pred_pts_bbox, gt_pts_bbox, out_dir, img=None, in_simu=False):
    mmcv.mkdir_or_exist(out_dir)
    
    cam2img = gt_pts_bbox['cam2img']
    front_img = img
    if img is None:
        front_img = mmcv.imread(gt_pts_bbox['imgpath']) # TODO
    front_img = create_front(pred_pts_bbox, gt_pts_bbox, front_img, cam2img)
    bev_img = create_bev(pred_pts_bbox, gt_pts_bbox)
    collide_img = create_collide_bev(pred_pts_bbox=pred_pts_bbox, 
                                     gt_pts_bbox=dict(), # 没有传入预测的box和wp所以是也是none
                                     only_box_for_col_det=dict(
                                        front=0, # wp往上面延伸多少米
                                        width=2, # 画wp线段的时候宽度是2米，在里面会乘以像素大小
                                        forshow=True,
                                     ))
    # fut_box_img = create_fut_bev(pred_pts_bbox, gt_pts_bbox)
    fut_box_img = None
    
    args = []
    if collide_img is not None:
        args.append(collide_img)
    if bev_img is not None:
        args.append(bev_img)
    if fut_box_img is not None:
        args.append(fut_box_img)
    bev_img = np.concatenate(args, axis=1)

    h, w, _ = front_img.shape
    bh, bw, _ = bev_img.shape
    all_img = np.zeros((h+bh,max(w,bw),3)) # 黑色背景
    # all_img = np.full(shape=(h+s,max(w,s),3),fill_value=255) # 白色背景
    all_img[:h,:w] = front_img
    all_img[h:h+bh,(w-bw)//2:(w-bw)//2+bw] = bev_img
    
    # if not in_simu:
    #     front_img = all_img
    #     bev_img = pred_pts_bbox['attnmap'] # 没问题
    #     bev_img = bev_img[...,None]*255
    #     # a,b,c = n_bev_img.shape
    #     # bev_img = np.array(a*3,b*3,c)
    #     plt.figure(figsize=(8,1))
    #     plt.imshow(bev_img)
    #     plt.axis('off')
    #     plt.savefig('./a.png', bbox_inches='tight')
    #     plt.close()
    #     bev_img=mmcv.imread('./a.png')
        
    #     h, w, _ = front_img.shape
    #     bh, bw, _ = bev_img.shape
    #     all_img = np.zeros((h+bh,max(w,bw),3)) # 黑色背景
    #     # all_img = np.full(shape=(h+s,max(w,s),3),fill_value=255) # 白色背景
    #     all_img[:h,:w] = front_img
    #     all_img[h:h+bh,(w-bw)//2:(w-bw)//2+bw] = bev_img
    
    mmcv.imwrite(all_img, f'{out_dir}/{index:05d}.png')

def output_to_nusc_box(detection, with_velocity=True):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d']#.numpy()
    labels = detection['labels_3d']#.numpy()

    box_gravity_center = box3d.gravity_center#.numpy() tensor([10.2103, 30.8526,  0.1760]) 应该是相对位置
    box_dims = box3d.dims#.numpy() tensor([0.6808, 0.6693, 1.7621]) 也就是3:6的部分
    box_yaw = box3d.yaw#.numpy() -1.1953
    
    if isinstance(scores,torch.Tensor):
        scores = scores.numpy()
        labels = labels.numpy()
    
    if isinstance(box_gravity_center,torch.Tensor):
        box_gravity_center = box_gravity_center.numpy()
        box_dims = box_dims.numpy()
        box_yaw = box_yaw.numpy()

    # our LiDAR coordinate system -> nuScenes box coordinate system
    nus_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if with_velocity:
            velocity = (*box3d.tensor[i, 7:9], 0.0)#tensor([-0.4399, -1.0331])
        else:
            velocity = (0, 0, 0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],#tensor([25.3113, 17.3637, -0.2196])
            nus_box_dims[i],#tensor([1.8566, 1.6064, 1.7794])
            quat,#Quaternion(0.13915480322897048, 0.0, 0.0, 0.9902706401475844)
            label=labels[i],#tensor(8)
            score=scores[i],#tensor(0.1225)
            velocity=velocity)#(tensor(0.0029), tensor(-0.0073), 0.0)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:#label: 8, score: 0.85, xyz: [10.21, 30.85, 0.18], wlh: [0.67, 0.68, 1.76], rot axis: [0.00, 0.00, 1.00], ang(degrees): 25.91, ang(rad): 0.45, vel: -0.44, -1.03, 0.00, name: None, token: None
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # label: 8, score: 0.85, xyz: [31.81, -10.22, 0.56], wlh: [0.67, 0.68, 1.76], rot axis: [-0.02, 0.04, -1.00], ang(degrees): 64.17, ang(rad): 1.12, vel: -1.03, 0.44, 0.05, name: None, token: None
        # filter det in ego.
        cls_range_map = eval_configs.class_range#{'car': 50, 'truck': 50, 'bus': 50, 'trailer': 50, 'construction_vehicle': 50, 'pedestrian': 40, 'motorcycle': 40, 'bicycle': 40, 'traffic_cone': 30, 'barrier': 30}
        radius = np.linalg.norm(box.center[:2], 2)#33.4105223152271
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        # label: 8, score: 0.85, xyz: [623.10, 1623.23, -0.02], wlh: [0.67, 0.68, 1.76], rot axis: [0.01, -0.04, 1.00], ang(degrees): -92.91, ang(rad): -1.62, vel: -0.69, 0.88, 0.07, name: None, token: None
        box_list.append(box)
    return box_list



def filter_invisible(gt_bboxes_3d, img_shape, mat):
    points = gt_bboxes_3d.tensor.numpy()
    num_points = points.shape[0]

    shape = (img_shape[0], img_shape[1])#(256, 928)hw

    pts_4d = np.concatenate([points[:, :3], np.ones((num_points, 1))], axis=-1)
    pts_2d = pts_4d @ mat.T

    valid_mask = (pts_2d[..., 2] > 1e-5)
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e6)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    pts_2d[:, 0] /= shape[1]
    pts_2d[:, 1] /= shape[0]
    valid_mask = (valid_mask & (pts_2d[..., 0] > -1.0) 
                             & (pts_2d[..., 0] < 1.0) 
                             & (pts_2d[..., 1] > -1.0) 
                             & (pts_2d[..., 1] < 1.0))
    return valid_mask

def convert(box, rt_mat=None, with_yaw=True):

    is_numpy = isinstance(box, np.ndarray)
    is_Instance3DBoxes = isinstance(box, BaseInstance3DBoxes)
    single_box = isinstance(box, (list, tuple))
    if single_box:
        assert len(box) >= 7, (
            'Box3DMode.convert takes either a k-tuple/list or '
            'an Nxk array/tensor, where k >= 7')
        arr = torch.tensor(box)[None, :]
    else:
        # avoid modifying the input box
        if is_numpy:
            arr = torch.from_numpy(np.asarray(box)).clone()
        elif is_Instance3DBoxes:
            arr = box.tensor.clone()
        else:
            arr = box.clone()

    if is_Instance3DBoxes:
        with_yaw = box.with_yaw

    # convert box from `src` mode to `dst` mode.
    x_size, y_size, z_size = arr[..., 3:4], arr[..., 4:5], arr[..., 5:6]
    yaw = arr[..., 6:7]

    xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
    if with_yaw:
        # yaw = -yaw - np.pi / 2
        yaw = limit_period(yaw, period=np.pi * 2)

    if not isinstance(rt_mat, torch.Tensor):
        rt_mat = arr.new_tensor(rt_mat)
    if rt_mat.size(1) == 4:
        extended_xyz = torch.cat(
            [arr[..., :3], arr.new_ones(arr.size(0), 1)], dim=-1)
        xyz = extended_xyz @ rt_mat.t()
    else:
        xyz = arr[..., :3] @ rt_mat.t()

    remains = arr[..., 7:]
    arr = torch.cat([xyz[..., :3], xyz_size, yaw, remains], dim=-1)

    return LiDARInstance3DBoxes(arr, box_dim=arr.size(-1), with_yaw=with_yaw)

def rel2s3(string):
    if string.startswith('output/PlanT_data_1'):
        return f"s3://tr_plan_hrz/{'/'.join(string.split('/')[1:])}"
    elif string.startswith('output/PlanT_val_1/coke_s3_dataset/Routes_l6_dataset'):
        return f"s3://tr_plan_hrz/l6_dataset/Routes_l6_dataset/{'/'.join(string.split('/')[4:])}"
    return f"s3://tr_plan_hrz/{'/'.join(string.split('/')[2:])}"