import torch

from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from mmdet3d.core import bbox3d2result
import numpy as np
from collections import deque
import torch.nn.functional as F
from diskcache import Cache
import time

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
class Detr3D_gtgru(MVXTwoStageDetector):
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
                 frozen=False):
        super(Detr3D_gtgru,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

        # PID controller
        self.turn_controller = PIDController(K_P=0.9, K_I=0.75, K_D=0.3, n=20)
        self.speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=20)
        self.inited = False
        self.frozen = frozen

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
                      gt_bboxes_ignore=None):
        pred_wp = self.pts_bbox_head.forward_gtgru(img_metas=img_metas,gt_bboxes_3d=gt_bboxes_3d,gt_labels_3d=gt_labels_3d)
        loss = self.pts_bbox_head.loss_gru(pred_wp, img_metas)
        return loss

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats
        
    def extract_img_feat(self, img, img_metas):
        # t1 = time.time()
        if self.frozen and not self.inited:
            self.inited = True
            self.img_backbone.eval()
            for k in self.img_backbone.parameters():
                k.requires_grad = False
            self.img_neck.eval()
            for k in self.img_neck.parameters():
                k.requires_grad = False
            # self.shared_dict = Cache(directory='./cache_feat')# ,size_limit=int(768 * 1024 ** 3))

        # if self.frozen:
        #     ids = [t['sample_idx'] for t in img_metas]
        #     cached = True
        #     for idx in ids:
        #         if idx not in self.shared_dict:
        #             cached = False
        #             continue
        #     if cached:
        #         img_feats_reshaped = []
        #         for lvl in range(4):
        #             img_feats_reshaped.append([])
        #         for idx in ids:
        #             for lvl in range(4):
        #                 img_feats_reshaped[lvl].append(self.shared_dict[idx][lvl])
        #         for lvl in range(4):
        #             img_feats_reshaped[lvl] = torch.stack(img_feats_reshaped[lvl],dim=0)
        #         t2 = time.time()
        #         print(f"> cached time use {(t2-t1):.4f} s")
        #         return img_feats_reshaped
            
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                # img.squeeze_(0)
                img = img.squeeze(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            # tgm = time.time()
            img_feats = self.img_backbone(img)
            # tbk = time.time()
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        
        # if self.frozen:
        #     for i in range(B):
        #         item = []
        #         for lvl in range(4):
        #             item.append(img_feats_reshaped[lvl][i,...])
        #         self.shared_dict[ids[i]] = item

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
            # tnk = time.time()
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        
        # if self.frozen:
        #     for i in range(B):
        #         item = []
        #         for lvl in range(4):
        #             item.append(img_feats_reshaped[lvl][i,...])
        #         self.shared_dict[ids[i]] = item
        # t2 = time.time()
        # print(f">> uncached use {(t2-t1):.4f} s")
        # print(f">> extract_feat: grid_mask:{tgm-t1:.4f}s backbone:{tbk-tgm:.4f}s neck:{tnk-tbk:.4f}s ",end='')
        return img_feats_reshaped

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        # t1 = time.time()
        outs = self.pts_bbox_head(pts_feats, img_metas)
        # t2 = time.time()
        loss_inputs = [img_metas, gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        # t3 = time.time()
        # print(f"head:{(t2-t1):.4f}s loss:{(t3-t2):.4f}s ",end='')
        return losses, outs

    @torch.no_grad()
    def forward_test(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      img=None,
                      **kwargs):
        # img_feats = self.extract_feat(img=img, img_metas=img_metas)
        # losses, outs = self.forward_pts_train(img_feats, gt_bboxes_3d,
        #                                     gt_labels_3d, img_metas,
        #                                     gt_bboxes_ignore=None)
        # # 所以展示的loss是多层的均值
        # bbox_results = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=None) # [pts_bbox,...]
        # ret_list = [dict() for i in range(len(img_metas))]
        # for result_dict, pts_bbox in zip(ret_list, bbox_results):
        #     result_dict['pts_bbox'] = pts_bbox
        #     result_dict['loss'] = {k:losses[k].cpu().numpy() for k in losses}

        pred_wp = self.pts_bbox_head.forward_gtgru(img_metas=img_metas,gt_bboxes_3d=gt_bboxes_3d,gt_labels_3d=gt_labels_3d)
        loss = self.pts_bbox_head.loss_gru(pred_wp, img_metas)
        result_dict = {}
        result_dict['loss'] = {k:loss[k].detach().cpu().numpy() for k in loss}
        result_dict['wp'] = pred_wp.detach().cpu().numpy()
        return [result_dict]

    def forward_test_nus(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
    
    def simple_test(self, img_metas, img=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
        
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
