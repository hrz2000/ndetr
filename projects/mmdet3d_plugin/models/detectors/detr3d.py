import torch

from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from mmdet3d.core import bbox3d2result
import numpy as np
from collections import deque
import torch.nn.functional as F
import torch.nn as nn
from diskcache import Cache
import time
import copy

def get_k():
    K = np.identity(3)
    width= 900
    height= 256
    fov= 100
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    f = width / (2.0 * np.tan(fov * np.pi / 360.0))
    K[0, 0] = f
    K[1, 1] = f
    proj_mat_expanded = np.identity(4)
    proj_mat_expanded[:3, :3] = K
    K = proj_mat_expanded

    F = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,0]])

    vl2cam = np.eye(4)
    vl2cam[0,3] = 2.8
    vl2cam[1,3] = 0
    vl2cam[2,3] = 0.5
    cam2vl = np.linalg.inv(vl2cam)

    cam2vl_r = cam2vl[:3,:3]
    cam2vl_t = cam2vl[:3,-1]

    # cam2img = K @ F @ vl2cam
    return K @ F, vl2cam, cam2vl_r, cam2vl_t, cam2vl # 这个前后之间有差距和cam2vl[:3,-1]一样

@DETECTORS.register_module()
class Detr3D(MVXTwoStageDetector):
    """Detr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 his_withgrad=True,
                 temporal=None,
                 use_fv=True,
                 use_flatten_feat=False,
                 **kwargs):
        super(Detr3D,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.use_flatten_feat = use_flatten_feat
        if self.use_flatten_feat:
            if img_backbone['depth'] <= 34:
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512, 1000)
            else:
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512*4, 1000)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.his_withgrad = his_withgrad

        # PID controller
        self.turn_controller = PIDController(K_P=0.9, K_I=0.75, K_D=0.3, n=20)
        self.speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=20)
        self.pred_len = 4
        self.temporal = temporal
        self.use_fv = use_fv

        if self.temporal == 'bevformer':
            pass
        elif self.temporal == 'gruinfer':
            pass
        elif self.temporal == 'mutr3d':
            pass
        elif self.temporal == None:
            pass
        else:
            assert False, self.temporal

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      img=None,
                      fut_gt_bboxes_3d_list=None,
                      fut_gt_labels_3d_list=None,
                      gt_bboxes_ignore=None,
                      in_test=False,
                      **kawrgs):
        if self.temporal == None:
            if self.use_fv == True:
                img_feats, flatten_feat = self.extract_feat(img=img, img_metas=img_metas)
            else:
                img_feats = None
            pts_ret = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d, img_metas, flatten_feat=flatten_feat)
            outs = pts_ret['outs']
            losses = pts_ret['losses']
            
        elif self.temporal == 'bevformer': # 不保存前面几帧的梯度
            len_queue = img.size(1) # 4, 包含间隔1.5s的历史信息
            prev_img = img[:, :-1, ...] # torch.Size([bs=2, his=3, cam=1, 3, 256, 928])
            img = img[:, -1, ...]

            prev_img_metas = copy.deepcopy(img_metas) # 多了当前的metas不要紧
            if self.his_withgrad:
                prev_query, his_loss_dict = self.obtain_history_query_withgrad(prev_img, prev_img_metas)
            else:
                prev_query, his_loss_dict = self.obtain_history_query_nograd(prev_img, prev_img_metas)

            img_metas = [each[len_queue-1] for each in img_metas] # 这里img_metas已经变成了每个queue的最后一个
            img_feats = self.extract_feat(img=img, img_metas=img_metas)
            pts_ret = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d, img_metas, prev_query=prev_query)
            outs = pts_ret['outs']
            losses = pts_ret['losses']
            losses.update(his_loss_dict)
            
        else:
            pass
    
        if in_test:
            return losses, outs, img_metas
        return losses

    @torch.no_grad()
    def forward_test(self, **kwargs):
        losses, outs, img_metas = self.forward_train(in_test=True, **kwargs)
        bbox_results = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=None) # [pts_bbox,...]
        ret_list = [dict() for i in range(len(img_metas))]
        from projects.configs.detr3d.new.debug import debug
        debug = False
        if debug:
            for bid, (result_dict, pts_bbox, gt_bbox, gt_labels) in enumerate(zip(ret_list, bbox_results, kwargs['gt_bboxes_3d'], kwargs['gt_labels_3d'])):
                gt_pts_bbox = dict()
                gt_pts_bbox['boxes_3d'] = gt_bbox.to('cpu')
                gt_pts_bbox['labels_3d'] = gt_labels.cpu()
                gt_pts_bbox['scores_3d'] = np.ones_like(gt_labels.cpu())
                gt_pts_bbox['img_metas'] = kwargs['img_metas'][bid]
                result_dict['pts_bbox'] = gt_pts_bbox # 里面是字典包含boxes_3d等信息
                result_dict['loss'] = {k:losses[k].cpu().numpy() for k in losses}
        else:
            for result_dict, pts_bbox in zip(ret_list, bbox_results):
                result_dict['pts_bbox'] = pts_bbox # 里面是字典包含boxes_3d等信息
                result_dict['loss'] = {k:losses[k].cpu().numpy() for k in losses}
                # result_dict['loss'] = {k:losses[k] for k in losses} # TODO
        return ret_list

    def loss_add_prefix(self, inp_loss_dict, prefix=''):
        loss_dict = {f"{prefix}.{k}" : v for k, v in inp_loss_dict.items()}
        return loss_dict
    
    def obtain_history_query_nograd(self, imgs_queue, img_metas_list):
        self.eval()
        with torch.no_grad():
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape # torch.Size([bs=1, lenq=3, cam=1, 3, 256, 928])
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W) 
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            # [torch.Size([bs=1, lenq=3, cam=1, 256, 32, 116])]
            prev_query = None
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list] # 只拿出每个尺度的相应queue_idx
                # [torch.Size([bs=1, cam=1, 256, h=32, w=116]), torch.Size([1, 1, 256, 16, 58]), torch.Size([1, 1, 256, 8, 29]), torch.Size([1, 1, 256, 4, 15])] 这里面只包含一个时刻的img特征
                prev_query = self.pts_bbox_head(img_feats, img_metas, prev_query=prev_query, only_query=True)
            
            self.train()
            return prev_query, dict()

    def obtain_history_query_withgrad(self, imgs_queue, img_metas_list):
        bs, len_queue, num_cams, C, H, W = imgs_queue.shape # torch.Size([bs=1, lenq=3, cam=1, 3, 256, 928])
        imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W) 
        img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
        # [torch.Size([bs=1, lenq=3, cam=1, 256, 32, 116])]
        prev_query = None
        his_losses = {}
        for i in range(len_queue):
            img_metas = [each[i] for each in img_metas_list] # 这个帧的batch_size个img_metas列表
            img_feats = [each_scale[:, i] for each_scale in img_feats_list] # 只拿出每个尺度的相应queue_idx
            # [torch.Size([bs=1, cam=1, 256, h=32, w=116]), torch.Size([1, 1, 256, 16, 58]), torch.Size([1, 1, 256, 8, 29]), torch.Size([1, 1, 256, 4, 15])] 这里面只包含一个时刻的img特征
            outs, hs = self.pts_bbox_head(img_feats, img_metas, prev_query=prev_query, only_query=False)
            prev_query = hs[-1]
            gt_labels_3d = [t['gt_labels_3d'].to('cuda') for t in img_metas] # 都是list
            gt_bboxes_3d = [t['gt_bboxes_3d'] for t in img_metas]
            loss_inputs = [img_metas, gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs)
            his_losses.update(self.loss_add_prefix(losses, f'his_{i}'))
        return prev_query, his_losses
            
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # update real input shape of each single img
            # for img_meta in img_metas: # 修改后是[[]]所以不行
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                # img.squeeze_(0)
                img = img.squeeze(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if self.use_flatten_feat:
                assert len_queue == None, 'unimpletment'
                x_layer4 = img_feats[-1]
                BN, C, H, W = x_layer4.size()
                x_layer4 = x_layer4.view(B, int(BN / B), C, H, W) # 从每个相机计算是不太对的
                x_layer4 = x_layer4[:,0] # fv
                x = self.avgpool(x_layer4)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                flatten_feat = x
            else:
                flatten_feat = None
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for idx, img_feat in enumerate(img_feats):
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped, flatten_feat

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats, flatten_feat = self.extract_img_feat(img, img_metas, len_queue=len_queue) # 不经过backbone的特征
        
        return img_feats, flatten_feat

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          flatten_feat=None,
                          prev_query=None,
                          gt_bboxes_ignore=None):
        outs, hs = self.pts_bbox_head(pts_feats, img_metas, prev_query=prev_query, flatten_feat=flatten_feat)
        # outs['speed'] = pred_speed
        if gt_bboxes_3d is not None:
            loss_inputs = [img_metas, gt_bboxes_3d, gt_labels_3d, outs]
            losses, new_gt_idxs_list_layers = self.pts_bbox_head.loss(*loss_inputs)
            outs['new_gt_idxs_list_layers'] = new_gt_idxs_list_layers ## 放到这里了
            outs['all_'] = new_gt_idxs_list_layers ## 放到这里了
            # outs['new_gt_idxs_list_layers'] = new_gt_idxs_list_layers ## 放到这里了
        else:
            losses = {}
        return dict(
            hs=hs,
            outs=outs,
            losses=losses,
        )
        
    def show_results(self,
                    data,
                    result,
                    out_dir=None,
                    show=None,
                    score_thr=None):
        pass

    def control_pid(self, waypoints, velocity, is_stuck=False):
        """Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            velocity (tensor): speedometer input
        """
        # assert waypoints.size(0) == 1
        waypoints = waypoints.data.cpu().numpy()
        # when training we transform the waypoints to lidar coordinate, so we need to change is back when control
        # waypoints[:, 0] += 1.3
        # 预测是在ego坐标系下，那么这里需要变到lidar坐标系

        speed = velocity[0].data.cpu().numpy()

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        if is_stuck:
            desired_speed = np.array(4.0) # default speed of 14.4 km/h

        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0
        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90
        if brake:
            angle = 0.0
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        return steer, throttle, brake

class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative
