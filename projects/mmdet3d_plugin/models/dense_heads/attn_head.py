from cmath import polar
import copy

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32
                        
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
# from mmdet3d.core import bbox3d2result
import numpy as np
import math
import mmcv
import os
from projects.mmdet3d_plugin.datasets.vis_tools import create_bev, create_collide_bev
from mmcv.parallel import DataContainer as DC
import time
from projects.mmdet3d_plugin.models.utils.detr3d_ca import CrossAttn
from mmcv.runner.base_module import BaseModule

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(
            masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

class CtnetHead(nn.Module):
    def __init__(self, heads, channels_in, train_cfg=None, test_cfg=None, down_ratio=4, final_kernel=1, head_conv=256,
                 branch_layers=0):
        super(CtnetHead, self).__init__()

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0: # 64
                fc = nn.Sequential(
                    nn.Conv2d(channels_in, head_conv, # channels_in=64
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=final_kernel, stride=1,
                              padding=final_kernel // 2, bias=True)) # channel dim -> 1|2
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channels_in, classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
        return z

    def init_weights(self):
        # ctnet_head will init weights during building
        pass

class Comfort():
    def __init__(self):
        self.c_lat_acc = 3 # m/s2
        self.c_lon_acc = 3 # m/s2
        self.c_jerk = 1 # m/s3

        self.factor = 0.1

    def __call__(self, trajs):
        '''
        trajs: torch.Tensor<float> (B, N, n_future, 2)
        '''
        B, N, n_future, _ = trajs.shape
        lateral_velocity = torch.zeros((B,N,n_future), device=trajs.device)
        longitudinal_velocity = torch.zeros((B, N, n_future), device=trajs.device)
        lateral_acc = torch.zeros((B,N,n_future), device=trajs.device)
        longitudinal_acc = torch.zeros((B,N,n_future), device=trajs.device)
        for i in range(n_future):
            if i == 0:
                lateral_velocity[:,:,i] = trajs[:,:,i,0] / 0.5
                longitudinal_velocity[:,:,i] = trajs[:,:,i,1] / 0.5
            else:
                lateral_velocity[:,:,i] = (trajs[:,:,i,0] - trajs[:,:,i-1,0]) / 0.5
                longitudinal_velocity[:,:,i] = (trajs[:,:,i,1] - trajs[:,:,i-1,1]) / 0.5
        for i in range(1, n_future):
            lateral_acc[:,:,i] = (lateral_velocity[:,:,i] - lateral_velocity[:,:,i-1]) / 0.5
            longitudinal_acc[:,:,i] = (longitudinal_velocity[:,:,i] - longitudinal_velocity[:,:,i-1]) / 0.5
        lateral_acc, _ = torch.abs(lateral_acc).max(dim=-1)
        longitudinal_acc, _ = torch.abs(longitudinal_acc).max(dim=-1)
        # v^2 - v_0^2 = 2ax
        # lateral_acc = (lateral_velocity[:,:,-1] ** 2 - lateral_velocity[:,:,0] ** 2) / (2 * (trajs[:,:,-1,0] - trajs[:,:,0,0]))
        # longitudinal_acc = (longitudinal_velocity[:,:,-1] ** 2 - longitudinal_velocity[:,:,0] ** 2) / (2 * (trajs[:,:,-1,1] - trajs[:,:,0,1]))
        # index = torch.isnan(lateral_acc)
        # lateral_acc[index] = 0.0
        # index = torch.isnan(longitudinal_acc)
        # longitudinal_acc[index] = 0.0

        # jerk
        ego_velocity = torch.zeros((B, N, n_future), device=trajs.device)
        ego_acc = torch.zeros((B,N,n_future), device=trajs.device)
        ego_jerk = torch.zeros((B,N,n_future), device=trajs.device)
        for i in range(n_future):
            if i == 0:
                ego_velocity[:, :, i] = torch.sqrt((trajs[:, :, i] ** 2).sum(dim=-1)) / 0.5
            else:
                ego_velocity[:, :, i] = torch.sqrt(((trajs[:, :, i] - trajs[:, :, i - 1]) ** 2).sum(dim=-1)) / 0.5
        for i in range(1, n_future):
            ego_acc[:,:,i] = (ego_velocity[:,:,i] - ego_velocity[:,:,i-1]) / 0.5
        for i in range(2, n_future):
            ego_jerk[:,:,i] = (ego_acc[:,:,i] - ego_acc[:,:,i-1]) / 0.5
        ego_jerk,_ = torch.abs(ego_jerk).max(dim=-1)

        subcost = torch.zeros((B, N), device=trajs.device)

        lateral_acc = torch.clamp(torch.abs(lateral_acc) - self.c_lat_acc, 0,30)
        subcost += lateral_acc ** 2
        longitudinal_acc = torch.clamp(torch.abs(longitudinal_acc) - self.c_lon_acc, 0, 30)
        subcost += longitudinal_acc ** 2
        ego_jerk = torch.clamp(torch.abs(ego_jerk) - self.c_jerk, 0, 20)
        subcost += ego_jerk ** 2

        return subcost * self.factor

class Progress():
    def __init__(self):
        self.factor = 0.5

    def __call__(self, trajs, target_points):
        '''
        trajs: torch.Tensor<float> (B, N, n_future, 2)
        target_points: torch.Tensor<float> (B, 2)
        '''
        B, N, n_future, _ = trajs.shape
        subcost1, _ = trajs[:,:,:,1].max(dim=-1) # 选择前后坐标最靠上的值(在4个未来时刻中取max)

        if target_points.sum() < 0.5:
            subcost2 = 0
        else:
            trajs = trajs[:,:,-1] # (B, N, 2)
            target_points = target_points.unsqueeze(1)
            subcost2 = ((trajs - target_points) ** 2).sum(dim=-1)

        return (subcost2 - subcost1) * self.factor

class Wp_grurefine(nn.Module):
    def __init__(self, head):
        super(Wp_grurefine, self).__init__() # loss_cfg在里面
        self.pred_len = head.pred_len
        self.use_cmd = head.use_cmd
        self.use_proj = head.use_proj
        self.gru_use_box = head.gru_use_box
        self.wp_refine_input_last = head.wp_refine_input_last
        n_embd = head.transformer.embed_dims
        query_size = 64
        hidden_size = 64+1 # light
        input_size = 2 + 2 # cp, tp
        if self.use_cmd:
            hidden_size += 1 # command
        if self.wp_refine_input_last:
            input_size += 2
        if self.gru_use_box:
            input_size += self.gru_use_box * 10 # 9+1
            # input_size += self.gru_use_box * 7 # 9dim+1scores
            # 之前也是*10，是reg的输出；现在是9+1
        self.wp_head = nn.Linear(n_embd, query_size)
        self.wp_decoder = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.wp_relu = nn.ReLU()
        self.wp_output = nn.Linear(hidden_size, 2)
        if self.use_proj:
            self.wp_proj = nn.Linear(64+1+2+2, 2)#隐藏维度，红绿灯，tp，当前点

    def forward(self, x_last, cls_emb, tp_batch, light_batch, cmd_batch, batch_npred_bbox):
        if not self.gru_use_box:
            batch_npred_bbox = None
        
        bs = len(cls_emb)
        z = self.wp_head(cls_emb) # bs,256->64
        if self.use_cmd:
            z = torch.cat((z, light_batch, cmd_batch), 1) # torch.Size([bs, 64+1+1])
        else:
            z = torch.cat((z, light_batch), 1)
        output_wp = list()
        x = z.new_zeros(size=(z.shape[0], 2))

        for idx_npred in range(self.pred_len): # 4
            if self.use_proj:
                x_in = torch.cat([z, x, tp_batch], dim=1)
                dx = self.wp_proj(x_in)
                x = dx + x
                output_wp.append(x)
            else:
                x_in = torch.cat([x, tp_batch], dim=1) # bs,(2+2)
                if self.wp_refine_input_last:
                    x_in = torch.cat([x_in, x_last[:,idx_npred]], dim=1)
                if self.gru_use_box:
                    x_in = torch.cat([x_in, batch_npred_bbox[:, :, idx_npred].reshape(bs, -1)], dim=1) # bs, num_box, npred, 10d
                z = self.wp_decoder(x_in, z) # torch.Size([1, 65])
                dx = self.wp_output(z) # torch.Size([1, 2])
                x = dx + x
                output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1) # bs,4,2
        pred_wp[:, :, 0] -= 1.3 # 变成了lidar系
        return pred_wp

class Wp_refine(nn.Module):
    def __init__(self, head):
        self.num_reg_fcs = head.num_reg_fcs
        self.embed_dims = head.transformer.embed_dims
        super(Wp_refine, self).__init__() # loss_cfg在里面
        wp_branch = []
        for _ in range(self.num_reg_fcs):
            wp_branch.append(Linear(self.embed_dims, self.embed_dims))
            wp_branch.append(nn.ReLU())
        wp_branch.append(Linear(self.embed_dims, 8))
        self.wp_branch = nn.Sequential(*wp_branch)
    
    def forward(self, cls_emb, tp_batch=None, light_batch=None, cmd_batch=None, batch_npred_bbox=None):
        bs = len(cls_emb)
        return self.wp_branch(cls_emb).reshape(bs, 4, 2)
        
    # def init_weights(self):
    #     """Initialize weights of the DeformDETR head."""
    #     self.transformer.init_weights()
    #     if self.loss_cls.use_sigmoid:
    #         bias_init = bias_init_with_prob(0.01)
    #         for m in self.cls_branches:
    #             nn.init.constant_(m[-1].bias, bias_init)

class Penalty():
    def __init__(self, 
                use_route_penalty=False,
                use_collide_penalty=False,
                use_comfort_penalty=False,
                use_progress_penalty=False,
                 ):
        self.use_route_penalty=use_route_penalty
        self.use_collide_penalty=use_collide_penalty
        self.use_comfort_penalty=use_comfort_penalty
        self.use_progress_penalty=use_progress_penalty
        if self.use_comfort_penalty:
            self.comfort = Comfort()
        if self.use_progress_penalty:
            self.progress = Progress()
    
    def __call__(self, pred_wp, route_wp, iscollide, tp, losses, head):
        if self.use_route_penalty:
            losses_wp = F.l1_loss(pred_wp, route_wp, reduction='none').mean([1,2,3]) # 剩下不同layer了
            losses_wp = losses_wp.mean()
            losses.update(head.loss_add_prefix({"loss":losses_wp}, 'route_penalty'))
        
        if self.use_collide_penalty:
            expect_wp = pred_wp.clone() # torch.Size([6, 1, 4, 2])
            expect_wp[:,iscollide,...] = 0 
            losses_wp = F.l1_loss(pred_wp, expect_wp, reduction='none').mean([1,2,3])
            losses_wp = losses_wp.mean()
            losses.update(head.loss_add_prefix({"loss":losses_wp}, 'collide_penalty'))
        
        if self.use_comfort_penalty:
            # pred_wp torch.Size([layer=6, bs=2, npred=4, xy=2])
            trajs = pred_wp.clone().permute(1,0,2,3)
            loss = self.comfort(trajs).mean() # torch.Size([2, 6]) 2个bs，6个层的轨迹的舒适度
            losses.update(head.loss_add_prefix({"loss":loss}, 'comfort_penalty'))

        if self.use_progress_penalty:
            # pred_wp
            trajs = pred_wp.clone().permute(1,0,2,3) # (B,1,n_future)
            loss = self.progress(trajs, tp).mean() # torch.Size([2, 6]) 刚开始时候特别大
            losses.update(head.loss_add_prefix({"loss":loss}, 'progress_penalty'))

@HEADS.register_module()
class AttnHead(BaseModule):
    """Head of Detr3D. 
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """
    def __init__(self,
                 use_hdmap=False,
                 use_route=False,
                 use_heatmap=False,
                 **kwargs):
        super(AttnHead, self).__init__() # loss_cfg在里面
        self.use_hdmap = use_hdmap
        self.use_route = use_route
        self.use_heatmap = use_heatmap
        self.pred_len = 4
        self.embed_dims = 256 # feat/pos
        self.query_embedding = nn.Embedding(1, self.embed_dims * 2)
        self.fv_cross_attn = CrossAttn()
        self.meas_cross_attn = CrossAttn()
        self.hdmap_cross_attn = CrossAttn()
        
        if self.use_route:
            self.meas_linear = nn.Linear(4+6, 256) # no_PE
        else:
            self.meas_linear = nn.Linear(4, 256) # no_PE
        bev_size = 300
        self.bev_down_size = bev_size // 16
        self.bev_linear = nn.Linear(2*self.bev_down_size*self.bev_down_size, self.embed_dims)
        self.wp_decocer = nn.Linear(256, 8)
        
        
        in_channels = 256
        self.keypts_head_box = CtnetHead(
            heads=dict(hm=1),
            channels_in=in_channels,
            final_kernel=1,
            head_conv=in_channels)
        self.keypts_head_lane = CtnetHead(
            heads=dict(hm=1),
            channels_in=in_channels,
            final_kernel=1,
            head_conv=in_channels)
        

        # self.offset_head = CtnetHead(
        #     heads=dict(offset_map=2),
        #     channels_in=in_channels,
        #     final_kernel=1,
        #     head_conv=in_channels)

        # self.reg_head = CtnetHead(
        #     heads=dict(offset_map=2),
        #     channels_in=in_channels,
        #     final_kernel=1,
        #     head_conv=in_channels)
        
        self._init_layers()

    def _init_layers(self):
        pass
        
    def forward(self, mlvl_feats, img_metas, prev_query=None, only_query=False):
        tp_batch = torch.tensor(np.stack([m['plan']['tp'] for m in img_metas],axis=0)).to('cuda').to(torch.float32) # (bs,2)
        light_batch = torch.tensor(np.stack([m['plan']['light'] for m in img_metas],axis=0)).reshape(-1,1).to('cuda').to(torch.float32) # (bs,1)
        cmd_batch = torch.tensor(np.stack([m['plan']['command'] for m in img_metas],axis=0)).reshape(-1,1).to('cuda').to(torch.float32) # (bs,1)
        route_batch = torch.tensor(np.stack([t['plan']['route'][0] for t in img_metas])).to('cuda').to(torch.float32) # (bs, 6)
        bs = len(tp_batch)
        
        if self.use_route:
            navi_batch = torch.cat([tp_batch, light_batch, cmd_batch, route_batch], dim=-1)
        else:
            navi_batch = torch.cat([tp_batch, light_batch, cmd_batch], dim=-1) # bs,4
        navi_batch = self.meas_linear(navi_batch)
        navi_batch = navi_batch[None] # 1seq,bs,xx
        
        wp_query = self.query_embedding.weight
        wp_query = wp_query[None].repeat(1,bs,1) # 1seq,bs,xx
        query_pos, query = torch.split(wp_query, self.embed_dims, dim=-1)

        query, query_pos = self.meas_cross_attn(query=query, query_pos=query_pos, key=navi_batch, key_pos=None, use_fv_sinpe=False)
        
        if self.use_hdmap:
            bev_batch = np.stack([self.get_hdmap(t['topdown_path']) for t in img_metas])
            bev_batch = query.new_tensor(bev_batch)
            bev_batch = F.interpolate(
                bev_batch, size=(self.bev_down_size, self.bev_down_size), mode='bilinear', align_corners=False).flatten(1) 
            bev_batch = self.bev_linear(bev_batch)  # 2*18*18->256d
            bev_batch = bev_batch[None]
            query, query_pos = self.hdmap_cross_attn(query=query, query_pos=query_pos, key=bev_batch, key_pos=None, use_fv_sinpe=False)

        # mlvl_feats: 4lvl, bs, cam, channels, h, w
        if mlvl_feats is not None:
            fv_feat_lvl1 = mlvl_feats[0]
            fv_feat_cam0 = fv_feat_lvl1[:,0]
            fv_feat = fv_feat_cam0 # bs, c, h, w
            query, query_pos = self.fv_cross_attn(query=query, query_pos=query_pos, key=fv_feat, key_pos=None, use_fv_sinpe=True)
        
        wp = self.wp_decocer(query).reshape(bs, 4, 2)
        
        if self.use_heatmap:
            kpts_hm_box = self.keypts_head_box(fv_feat)['hm'] # 解码box
            kpts_hm_box = torch.clamp(kpts_hm_box.sigmoid(), min=1e-4, max=1 - 1e-4)
            
            kpts_hm_lane = self.keypts_head_lane(fv_feat)['hm'] # 解码lane
            kpts_hm_lane = torch.clamp(kpts_hm_lane.sigmoid(), min=1e-4, max=1 - 1e-4)
            
            # torch.Size([2, 1, 32, 116])
        else:
            kpts_hm_box = None
            kpts_hm_lane = None

        outs = dict(
            wp=wp,
            kpts_hm_box=kpts_hm_box,
            kpts_hm_lane=kpts_hm_lane
        )
        return outs, None
    
    def get_hdmap(self, x):
        return np.stack([mmcv.imread(x.replace('topdown','hdmap0'),'grayscale'), mmcv.imread(x.replace('topdown','hdmap1'),'grayscale')],axis=0)

    def get_box_batch(self, batch_pts_boxes):
        # 只根据最后一层解码出box
        box_batch = []
        for pts_bbox in batch_pts_boxes:
            scores = pts_bbox['scores_3d']
            coords = pts_bbox['boxes_3d'].tensor
            coords = torch.cat([coords, scores[...,None]], dim=-1) # scores表示可靠程度，新加的是0
            coords[:, 0] += 1.3 # lidar2ego, z没什么用
            # coords = coords[:,[0,1,3,4,6,7,8]] # 01 34 6 78
            num, cdim = coords.shape
            if num < self.gru_use_box:
                tmp = coords.new_zeros((self.gru_use_box, cdim))
                tmp[:num] = coords
                coords = tmp
            distance = torch.sqrt((coords[:,0])**2 + coords[:,1]**2)
            box_idx = distance.sort(dim=0)[1][:self.gru_use_box] # 升序
            box = coords[box_idx] # (3,10)
            box_batch.append(box)
        box_batch = torch.stack(box_batch).clone() # (bs, num, cdim)
        
        # 未来预测
        box_batch = box_batch[:,:,None].repeat(1,1,self.pred_len,1)
        if self.velo_update:
            # yaw其实是没用的，检测到的速度是x和y两个方向，分别进行调整就没问题了
            for i in range(self.pred_len):
                # import pdb;pdb.set_trace()
                box_batch[:,:,i,0] += (i+1) * 0.5 * box_batch[:,:,0,8] # 往前后方向的速度0.5s
                box_batch[:,:,i,1] += (i+1) * 0.5 * box_batch[:,:,0,7] # 往左右方向的速度0.5s
                # box_batch[:,:,i,0] += (i+1) * 0.5 * box_batch[:,:,0,7] # 往x方向的速度0.5s
                # box_batch[:,:,i,1] += (i+1) * 0.5 * box_batch[:,:,0,8]
                # box_batch[:,:,i,0] -= (i+1) * 0.5 * box_batch[:,:,0,8] # 往x方向的速度0.5s
                # box_batch[:,:,i,1] -= (i+1) * 0.5 * box_batch[:,:,0,7]
                # 0,1,2  3,4,5  6  7,8
                # 7代表的是左右方向，对应1位置
                # 8代表的是上下方向，对应0位置
                
        return box_batch
    
    # def queries2outs(self, queries):
    #     pass
    
    # def pred_waypoint_per_layer(self, cls_emb, tp_batch, light_batch, cmd_batch, batch_npred_bbox):
    #     if not self.gru_use_box:
    #         batch_npred_bbox = None
        
    #     bs = len(cls_emb)
    #     z = self.wp_head(cls_emb) # bs,256->64
    #     if self.use_cmd:
    #         z = torch.cat((z, light_batch, cmd_batch), 1) # torch.Size([bs, 64+1+1])
    #     else:
    #         z = torch.cat((z, light_batch), 1)
    #     output_wp = list()
    #     x = z.new_zeros(size=(z.shape[0], 2))

    #     for idx_npred in range(self.pred_len): # 4
    #         if self.use_proj:
    #             x_in = torch.cat([z, x, tp_batch], dim=1)
    #             dx = self.wp_proj(x_in)
    #             x = dx + x
    #             output_wp.append(x)
    #         else:
    #             x_in = torch.cat([x, tp_batch], dim=1) # bs,(2+2)
    #             if self.gru_use_box:
    #                 x_in = torch.cat([x_in, batch_npred_bbox[:, :, idx_npred].reshape(bs, -1)], dim=1) # bs, num_box, npred, 10d
    #             z = self.wp_decoder(x_in, z) # torch.Size([1, 65])
    #             dx = self.wp_output(z) # torch.Size([1, 2])
    #             x = dx + x
    #             output_wp.append(x)

    #     pred_wp = torch.stack(output_wp, dim=1) # bs,4,2
    #     pred_wp[:, :, 0] -= 1.3 # 变成了lidar系
    #     return pred_wp

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :9]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, 
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0) # 2个类别，但是会有0,1,2
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        loss_cls = loss_cls.squeeze()

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1)) # bs, n_q, 10
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range) # torch.Size([900, 9])->torch.Size([900, 10])

        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos) # torch.Size([900, 10])

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    def loss(self,
             img_metas,
             gt_bboxes_list,
             gt_labels_list,
             outs,
             gt_bboxes_ignore=None):
        
        #draw_umich_gaussian
        kpts_hm_box = outs['kpts_hm_box']
        kpts_hm_lane = outs['kpts_hm_lane']
        
        # wp_loss
        pred_wp = outs['wp'] # bs,4,2
        gt_wp = np.stack([m['plan']['wp'] for m in img_metas], axis=0)
        gt_wp = pred_wp.new_tensor(gt_wp)
        losses_wp = F.l1_loss(pred_wp, gt_wp, reduction='none').mean()
        loss_dict = {'wp_loss':losses_wp, 'wp':losses_wp}
        
        return loss_dict
    
    @force_fp32(apply_to=('preds_dicts'))
    def lossx(self,
             img_metas,
             gt_bboxes_list,
             gt_labels_list,
             outs,
             gt_bboxes_ignore=None):
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = outs['all_cls_scores']
        all_bbox_preds = outs['all_bbox_preds']
        refine_wp = outs['refine_wp']
        enc_cls_scores = outs['enc_cls_scores']
        enc_bbox_preds = outs['enc_bbox_preds']
        pred_wp = outs['all_wp_preds']
        iscollide = outs['iscollide']
        route_wp = outs['route_wp']
        tp = outs['tp']

        num_dec_layers = len(all_cls_scores)
        if isinstance(gt_labels_list[0], DC):
            gt_bboxes_list = [t.data for t in gt_bboxes_list]
            gt_labels_list = [t.data.to('cuda') for t in gt_labels_list]

        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        # 这里gt_bboxes_list不能是Lidarin，应该是batch个

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, 
            all_gt_bboxes_ignore_list)

        losses = {}
        
        losses_box_cls_dict = {}
        for i, loss in enumerate(losses_cls):
            losses_box_cls_dict[f'd{i}'] = loss
        losses.update(self.loss_add_prefix(losses_box_cls_dict, 'det_cls'))
        
        losses_box_reg_dict = {}
        for i, loss in enumerate(losses_bbox):
            losses_box_reg_dict[f'd{i}'] = loss
        losses.update(self.loss_add_prefix(losses_box_reg_dict, 'det_reg'))
        
        # wp_loss
        gt_wp = np.stack([m['plan']['wp'] for m in img_metas],axis=0)
        gt_wp = pred_wp.new_tensor(gt_wp)
        gt_wp = gt_wp.unsqueeze(0).repeat(len(pred_wp),1,1,1) # torch.Size([6, 2, 4, 2])
        
        losses_wp = F.l1_loss(pred_wp, gt_wp, reduction='none').mean([1,2,3]) # torch.Size([6, 1, 4, 2])
        losses_wp_dict = {}
        for i, loss in enumerate(losses_wp):
            losses_wp_dict[f'd{i}'] = loss
        losses.update(self.loss_add_prefix(losses_wp_dict, 'wp'))
            
        if self.wp_refine:
            losses_wp = F.l1_loss(refine_wp, gt_wp, reduction='none').mean([1,2,3])
            
            losses_refine_wp_dict = {}
            for i, loss in enumerate(losses_wp):
                losses_refine_wp_dict[f'd{i}'] = loss
            losses.update(self.loss_add_prefix(losses_refine_wp_dict, 'refine_wp'))
        
        self.penalty(pred_wp, route_wp, iscollide, tp, losses, self)
            
        return losses
    
    def loss_add_prefix(self, inp_loss_dict, prefix=''):
        if self.enable_uncertainty_loss_weight:
            if prefix == 'det_cls':
                loss_factor = 1 / torch.exp(self.det_cls_weight)
                loss_uncertainty = 0.5 * self.det_cls_weight
            elif prefix == 'det_reg':
                loss_factor = 1 / torch.exp(self.det_reg_weight)
                loss_uncertainty = 0.5 * self.det_reg_weight
            elif prefix == 'wp':
                loss_factor = 1 / torch.exp(self.plan_weight)
                loss_uncertainty = 0.5 * self.plan_weight
            elif prefix == 'refine_wp':
                loss_factor = 1 / torch.exp(self.refine_plan_weight)
                loss_uncertainty = 0.5 * self.refine_plan_weight
            elif prefix == 'route_penalty':
                loss_factor = 1 / torch.exp(self.route_penalty_weight)
                loss_uncertainty = 0.5 * self.route_penalty_weight
            elif prefix == 'collide_penalty':
                loss_factor = 1 / torch.exp(self.collide_penalty_weight)
                loss_uncertainty = 0.5 * self.collide_penalty_weight
            elif prefix == 'progress_penalty':
                loss_factor = 1 / torch.exp(self.collide_penalty_weight)
                loss_uncertainty = 0.5 * self.collide_penalty_weight
            elif prefix == 'comfort_penalty':
                loss_factor = 1 / torch.exp(self.collide_penalty_weight)
                loss_uncertainty = 0.5 * self.collide_penalty_weight
            else:
                print(prefix)
                import pdb;pdb.set_trace()
                raise NotImplementedError
            loss_dict = {f"{prefix}.{k}_loss" : v*loss_factor for k, v in inp_loss_dict.items()}
            loss_dict[f"{prefix}.loss_uncertainty"] = loss_uncertainty
            if 'wp' in prefix: # 这个不会计算loss，只是直观展示
                loss_dict[prefix] = inp_loss_dict['d5']
        else:
            if prefix == 'det_cls':
                loss_factor = self.det_cls_weight
            elif prefix == 'det_reg':
                loss_factor = self.det_reg_weight
            elif prefix == 'wp':
                loss_factor = self.plan_weight
            elif prefix == 'refine_wp':
                loss_factor = self.refine_plan_weight
            elif prefix == 'route_penalty':
                loss_factor = self.route_penalty_weight
            elif prefix == 'collide_penalty':
                loss_factor = self.collide_penalty_weight
            elif prefix == 'progress_penalty':
                loss_factor = self.progress_penalty_weight
            elif prefix == 'comfort_penalty':
                loss_factor = self.comfort_penalty_weight
            else:
                raise NotImplementedError
            loss_dict = {f"{prefix}.{k}_loss" : v*loss_factor for k, v in inp_loss_dict.items()}
            if 'wp' in prefix: # 这个不会计算loss，只是直观展示
                loss_dict[prefix] = inp_loss_dict['d5']
        return loss_dict
    
    def get_route_wp(self, wp, route_batch, nlayer):
        # wp传入gt/pred_wp，注意gt_wp传进来之前需要进行repeat
        if len(wp.shape) == 3:
            wp = wp[None,...].repeat(nlayer,1,1,1)
        layer_batch_len_x = wp[:,:,:,0] + 1.3 # lidar到ego
        layer_batch_len_y = wp[:,:,:,1] # layer6,bs,pred4
        nlayer, bs, pred_len, _ = wp.shape

        bs, ndim = route_batch.shape
        route_batch = route_batch.reshape(1, bs, 1, ndim).repeat(nlayer, 1, pred_len, 1)
        batch_route_x = route_batch[:,:,:,0]
        batch_route_y = route_batch[:,:,:,1]
        batch_yaw = route_batch[:,:,:,2] # bs

        # route_expected_x = layer_batch_len_y**2 * np.tan(batch_yaw) / (2*batch_route_y)
        # route_expected_y = torch.tan(batch_yaw) / (2*batch_route_x) * layer_batch_len_x**2
        route_expected_y = (- batch_route_y / batch_route_x**2) * layer_batch_len_x**2
        
        # 将两侧调整为直线
        route_expected_y = torch.where(
            layer_batch_len_x >= batch_route_x, 
            -(torch.tan(batch_yaw)*(layer_batch_len_x-batch_route_x)+batch_route_y), 
            route_expected_y)
        route_expected_y = torch.where(
            layer_batch_len_x < 0, layer_batch_len_y, route_expected_y)
        route_wp = torch.stack([layer_batch_len_x, route_expected_y], dim=-1) # nlayer, bs, 4, 2
        route_wp[...,0] -= 1.3 # ego->lidar
        return route_wp

    def get_iscollide(self, batch_pts_boxes):
        # 仅考虑最后一层box和wp
        batch_bool = []
        for idx, pts_bbox in enumerate(batch_pts_boxes):
            img_and = create_collide_bev(pred_pts_bbox=pts_bbox, 
                                        gt_pts_bbox=dict(), 
                                        only_box_for_col_det=dict(
                                            front=0, # wp往上面延伸多少米
                                            width=2, # 画wp线段的时候宽度是2米，在里面会乘以像素大小
                                            forshow=False,
                                        ))
            colli = img_and.sum() > 0
            batch_bool.append(colli)
        return torch.tensor(batch_bool).to('cuda')

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, outs, img_metas, rescale=False, no_filter=False, for_aux_outs=False):
        # preds_dicts = self.bbox_coder.decode(outs, no_filter=no_filter)
        # [dict_keys(['bboxes', 'scores', 'labels']),...]
        wp = outs['wp'] # bs,4,2
        ret = [dict(
            attrs_3d=t.cpu(),
            img_metas=img_metas[i]) for i,t in enumerate(wp)]
        return ret
        # num_samples = len(preds_dicts)
        # ret_list = []
        # for i in range(num_samples):
        #     preds = preds_dicts[i]
        #     bboxes = preds['bboxes'] # torch.Size([300, 9])
        #     bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        #     bboxes = img_metas[i]['box_type_3d'](bboxes, 9)
        #     scores = preds['scores'] # torch.Size([300])
        #     labels = preds['labels'] # torch.Size([300])
        #     ret_list.append([bboxes, scores, labels])
        
        # if for_aux_outs:
        #     return [dict(
        #         boxes_3d=bboxes,
        #         scores_3d=scores,
        #         labels_3d=labels,
        #         # attrs_3d=outs['all_wp_preds'][-1][i] # 每个bs的最后一层
        #     ) for i, (bboxes, scores, labels) in enumerate(ret_list)]
        
        # bbox_results = []
        # for i, (bboxes, scores, labels) in enumerate(ret_list):
        #     attrs = None
        #     refine_wp = None
        #     route_wp = None
        #     iscollide = None
        #     fut_boxes = None
        #     if outs['all_wp_preds'] is not None:
        #         attrs=outs['all_wp_preds'][-1][i].cpu() # 最后一层的第i个batch
        #     if outs['refine_wp'] is not None:
        #         refine_wp=outs['refine_wp'][-1][i].cpu() # 最后一层的第i个batch
        #     if outs['route_wp'] is not None:
        #         route_wp=outs['route_wp'][-1][i].cpu()
        #     if outs['iscollide'] is not None:
        #         iscollide=outs['iscollide'][i].cpu() # bool
        #     if outs['fut_boxes'] is not None:
        #         fut_boxes=outs['fut_boxes'][i].cpu()
        #     pts_bbox = dict(
        #         boxes_3d=bboxes.to('cpu'),
        #         scores_3d=scores.cpu(),
        #         labels_3d=labels.cpu(),
        #         attrs_3d=attrs,
        #         refine_wp=refine_wp,
        #         route_wp=route_wp,
        #         iscollide=iscollide, 
        #         fut_boxes=fut_boxes,
        #         img_metas=img_metas[i] # 这个bs的img_metas
        #     )
        #     bbox_results.append(pts_bbox)
        # return bbox_results

def mybbox3d2result(bboxes, scores, labels, attrs=None, route=None, collide=None, collide_imgpath=None):
    result_dict = dict(
        boxes_3d=bboxes.to('cpu'),
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu())

    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()
    if route is not None:
        result_dict['route'] = route.cpu()
    if collide is not None:
        result_dict['collide'] = collide.cpu()
    if collide is not None:
        result_dict['collide_imgpath'] = collide_imgpath
    return result_dict