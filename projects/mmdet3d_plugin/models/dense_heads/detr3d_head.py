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
import mmcv
import os
from projects.mmdet3d_plugin.datasets.vis_tools import create_bev, create_collide_bev
from mmcv.parallel import DataContainer as DC
import time

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
            loss = self.comfort(trajs).mean()/5 # torch.Size([2, 6]) 2个bs，6个层的轨迹的舒适度
            losses.update(head.loss_add_prefix({"loss":loss}, 'comfort_penalty'))

        if self.use_progress_penalty:
            # pred_wp
            trajs = pred_wp.clone().permute(1,0,2,3) # (B,1,n_future)
            loss = self.progress(trajs, tp).mean()/300 # torch.Size([2, 6]) 刚开始时候特别大
            losses.update(head.loss_add_prefix({"loss":loss}, 'progress_penalty'))

@HEADS.register_module()
class Detr3DHead(DETRHead):
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
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=dict(),
                 bbox_coder=None,
                #  num_cls_fcs=2,
                 code_weights=None,
                use_proj=False,
                use_cmd=True,
                gru_use_box=0,
                # use_route_penalty=False,
                # use_collide_penalty=False,
                penalty_args=None,
                wp_refine = False,
                wp_refine_input_last = False,
                velo_update = False,
                use_gt_light = True,
                enable_uncertainty_loss_weight = True,
                loss_weights = None,
                 **kwargs
                 ):
        self.use_gt_light = use_gt_light
        self.wp_refine = wp_refine
        self.wp_refine_input_last = wp_refine_input_last
        assert self.wp_refine in ['gru', 'linear'] or self.wp_refine is None
        self.use_proj = use_proj
        self.use_cmd = use_cmd
        self.gru_use_box = gru_use_box
        self.penalty = Penalty(**penalty_args)
        self.penalty_args = penalty_args
        self.velo_update = velo_update
        self.enable_uncertainty_loss_weight = enable_uncertainty_loss_weight

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.pred_len = 4
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        # self.num_cls_fcs = num_cls_fcs - 1
        super(Detr3DHead, self).__init__(
            *args, transformer=transformer, **kwargs) # loss_cfg在里面
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.wp_query_pos = self.transformer.wp_query_pos
        self.route_query_pos = self.transformer.route_query_pos
        self.route_query_pos2 = self.transformer.route_query_pos2
        self.loss_weights = loss_weights
        self.extra_query = self.transformer.extra_query

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
        
        self.query_embedding = nn.Embedding(self.num_query,
                                            self.embed_dims * 2)
        # n_embd = self.transformer.embed_dims # 256
        
        if self.wp_refine == 'gru':
            self.wp_decoder = Wp_grurefine(self)
        elif self.wp_refine == 'linear':
            self.wp_decoder = Wp_refine(self)
        else:
            self.wp_decoder = Wp_grurefine(self)
            assert self.wp_refine == None
            
        if self.wp_refine is not None:
            if self.with_box_refine:
                self.wp_branches = _get_clones(self.wp_decoder, num_pred)
            else:
                self.wp_branches = nn.ModuleList([self.wp_decoder for _ in range(num_pred)])
        else:
            self.wp_branches = None
                
        if self.enable_uncertainty_loss_weight:
            self.det_cls_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.det_reg_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.plan_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            if self.wp_refine is not None:
                self.refine_plan_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            if self.penalty_args.use_route_penalty:
                self.route_penalty_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            if self.penalty_args.use_collide_penalty:
                self.collide_penalty_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            if self.penalty_args.use_comfort_penalty:
                self.comfort_penalty_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            if self.penalty_args.use_progress_penalty:
                self.progress_penalty_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        else:
            self.det_cls_weight = 1
            self.det_reg_weight = 1
            self.plan_weight = 1
            self.refine_plan_weight = 1
            
            self.route_penalty_weight = 1
            self.collide_penalty_weight = 1
            self.comfort_penalty_weight = 1
            self.progress_penalty_weight = 1

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, mlvl_feats, img_metas, prev_query=None, only_query=False):
        tp_batch = torch.tensor(np.stack([m['plan']['tp'] for m in img_metas],axis=0)).to('cuda').to(torch.float32)
        light_batch = torch.tensor(np.stack([m['plan']['light'] for m in img_metas],axis=0)).reshape(-1,1).to('cuda').to(torch.float32)
        cmd_batch = torch.tensor(np.stack([m['plan']['command'] for m in img_metas],axis=0)).reshape(-1,1).to('cuda').to(torch.float32)
        route_batch = torch.tensor(np.stack([t['plan']['route'][0] for t in img_metas])).to('cuda').to(torch.float32) # (-1, 6) # use_route_query
        if self.use_gt_light:
            pass
        else:
            light_batch[:] = 0 # test的时候也传入是0
        bs = len(tp_batch)
        cdim = 10
        batch_npred_bbox = torch.zeros((bs, self.gru_use_box, self.pred_len, cdim)).to('cuda')
        
        query_embeds = self.query_embedding.weight

        outputs = self.transformer(
            mlvl_feats=mlvl_feats,
            query_embed=query_embeds,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            wp_branches=self.wp_branches if self.with_box_refine else None,
            img_metas=img_metas,
            prev_query=prev_query,
            tp_batch=tp_batch,
            light_batch=light_batch,
            cmd_batch=cmd_batch,
            route_batch=route_batch,
            batch_npred_bbox=batch_npred_bbox,
        )

        hs, init_reference, inter_references, init_reference2, inter_references2, inter_attnmap = outputs
        hs = hs.permute(0, 2, 1, 3) # ->torch.Size([6, bs=6, num_queries, 256])
        
        if only_query:
            return hs[-1] # bs=6, num_queries, 256
        
        outputs_classes = []
        outputs_coords = []
        refine_wp_layers = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            
            if self.wp_refine:
                if lvl == 0:
                    reference2 = init_reference2
                else:
                    reference2 = inter_references2[lvl - 1]
                reference2 = inverse_sigmoid(reference2)
                tmp2 = self.wp_branches[lvl](reference2, hs[lvl][:,self.wp_query_pos], tp_batch, light_batch, cmd_batch, batch_npred_bbox=batch_npred_bbox)
                assert reference2.shape[-1] == 2
                tmp2[..., 0:2] += reference2[..., 0:2]
                tmp2[..., 0:2] = tmp2[..., 0:2].sigmoid()
                tmp2[..., 0:1] = (tmp2[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
                tmp2[..., 1:2] = (tmp2[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
                refined_wp = tmp2
                refine_wp_layers.append(refined_wp)

        self.extra_query = hs.shape[-2] % 10 ### 需要保证,如果我query1个或者2个没办法batch统一
        outputs_classes = torch.stack(outputs_classes)[:,:,self.extra_query:,:]
        outputs_coords = torch.stack(outputs_coords)[:,:,self.extra_query:,:] # torch.Size([6, 2, 50, 10])
        refine_wp_layers = torch.stack(refine_wp_layers) if self.wp_refine else [None]*len(outputs_classes) # torch.Size([6, 2, 4, 2])
        
        outs=dict(all_cls_scores=outputs_classes, all_bbox_preds=outputs_coords)
        batch_pts_boxes = self.get_bboxes(outs=outs, img_metas=img_metas, for_aux_outs=True)
        # 这里会获取一次box
        
        # 计算未来时刻的box位置、预测waypoint
        outputs_wps = []
        wp_embs = hs[:,:,self.wp_query_pos,:] # torch.Size([6, 2, 256])
        box_batch = self.get_box_batch(batch_pts_boxes) # 只拿最后一层预测来解码box, (bs, num_box, ndim=10)

        for lvl in range(wp_embs.shape[0]): # torch.Size([6, 2, 256])
            # outputs_wp = self.pred_waypoint_per_layer(
            #     wp_emb, tp_batch, light_batch, cmd_batch, box_batch)
            outputs_wp = self.wp_decoder(refine_wp_layers[lvl], wp_embs[lvl], tp_batch, light_batch, cmd_batch, box_batch)
            # 这里是用的外面的的wp_decoder，不重复            
            outputs_wps.append(outputs_wp)
        all_wp_preds = torch.stack(outputs_wps) # torch.Size([6, 1, 4, 2]), lidar系
        
        # 计算route_wp、iscollide
        for batch_id, pts_bbox in enumerate(batch_pts_boxes):
            pts_bbox['attrs_3d'] = all_wp_preds[-1,batch_id] # 只要最后一层
        
        route_wp = self.get_route_wp(all_wp_preds, route_batch, nlayer=len(wp_embs))
        iscollide = self.get_iscollide(batch_pts_boxes)
        
        outs = {
            'all_cls_scores': outputs_classes, # torch.Size([6, 1, 50, 2])
            'all_bbox_preds': outputs_coords,  # torch.Size([6, 1, 50, 10])
            'refine_wp': refine_wp_layers if self.wp_refine else None,  # torch.Size([6, bs=1, 4, 2])
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'all_wp_preds': all_wp_preds,
            'fut_boxes': box_batch, # (bs,3,npred,10d)
            'route_wp': route_wp,
            'iscollide': iscollide,
            'tp': tp_batch,
            'inter_attnmap': inter_attnmap,
            'attnmap': inter_attnmap,
        }
        # 还会根据这些信息再次获取box
        return outs, hs

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
                           gt_idxs,
                           gt_bboxes_ignore=None):

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result, new_gt_idxs = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_idxs, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds # tensor([ 5,  7, 20, 33, 35, 40, 42, 44], device='cuda:0')
        neg_inds = sampling_result.neg_inds # 剩余的

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
                pos_inds, neg_inds, new_gt_idxs)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_idxs_list,
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
         bbox_weights_list, pos_inds_list, neg_inds_list, new_gt_idxs_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_idxs_list, gt_bboxes_ignore_list) # 这些是不包含gt_id的信息
        # import pdb;pdb.set_trace()
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg), new_gt_idxs_list

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_idxs_list,
                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, gt_idxs_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg), new_gt_idxs_list = cls_reg_targets
        labels = torch.cat(labels_list, 0) # 2个类别，但是会有0,1,2
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0) # torch.Size([1600, 9]) 表示在9个属性上操作, 9个数行对应的value是什么
        bbox_weights = torch.cat(bbox_weights_list, 0) # torch.Size([1600=32*50, 10])
        # import pdb;pdb.set_trace()

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
        return loss_cls, loss_bbox, new_gt_idxs_list
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
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
        inter_attnmap = outs['inter_attnmap'] # torch.Size([32, 6, 8, 53, 53])

        num_dec_layers = len(all_cls_scores)
        if isinstance(gt_labels_list[0], DC):
            gt_bboxes_list = [t.data for t in gt_bboxes_list]
            gt_labels_list = [t.data.to('cuda') for t in gt_labels_list]

        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        # 这里gt_bboxes_list不能是Lidarin，应该是batch个
        gt_idxs_list = [t['gt_idxs'] for t in img_metas]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_idxs_list = [gt_idxs_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox, new_gt_idxs_list_layers = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_idxs_list,
            all_gt_bboxes_ignore_list) # 只在计算loss的时候才进行了new_gt_idxs的计算，怎么把它放到box里面里面

        losses = {}
        
        self.loss_update(losses, losses_cls, 'det_cls')
        self.loss_update(losses, losses_bbox, 'det_reg')
        
        # wp_loss
        gt_wp = np.stack([m['plan']['wp'] for m in img_metas],axis=0)
        gt_wp = pred_wp.new_tensor(gt_wp)
        gt_wp = gt_wp.unsqueeze(0).repeat(len(pred_wp),1,1,1) # torch.Size([6, 2, 4, 2])
        losses_wp = F.l1_loss(pred_wp, gt_wp, reduction='none').mean([1,2,3])
        self.loss_update(losses, losses_wp, 'wp')
            
        if self.wp_refine:
            losses_wp = F.l1_loss(refine_wp, gt_wp, reduction='none').mean([1,2,3])
            self.loss_update(losses, losses_wp, 'refine_wp')
        
        self.penalty(pred_wp, route_wp, iscollide, tp, losses, head=self)
        
        # 仿真的时候就不要再计算loss了
        if self.loss_weights.loss_attnmap != 0:
            # inter_attnmap = outs['inter_attnmap'] # torch.Size([32, 6, 8, 53, 53])
            for batch_id in range(len(inter_attnmap)):
                last_layer_batchi_gt = new_gt_idxs_list_layers[-1][batch_id].to(torch.int) # 每个gt对应着那些位置也是需要明确的
                attnmap = inter_attnmap[batch_id] # torch.Size([6, 8, 53, 53])
                attnmap = attnmap[-1:] # 最后一层的
                
                if 'attn_info' not in img_metas[batch_id]: # 仿真
                    break
                
                attn_info = img_metas[batch_id]['attn_info']
                gt_attnmap = attnmap.new_tensor(attn_info['attn_map']) # torch.Size([8, 8, 33, 33])
                gt_attnmap = gt_attnmap[-1:] # 最后一层的
                
                len_predmap = attnmap.shape[-1]
                len_gtmap = gt_attnmap.shape[-1]
                
                input_idx = attn_info['input_idx'] # 21#
                output_idx = attn_info['output_idx'] # 理论上inp和oup的idx是完全一样的
                assert len(input_idx) == len(output_idx)
                output_disappear = attn_info['output_disappear']
                
                len_box_route = len(input_idx)
                len_attn = 1 + len(input_idx) + 1
                # assert len_attn == len_gtmap # 咋input_idx比input_idx多呢，。。。。
                # 上面这个assert一定不对，因为有pad
                len_box = len(output_disappear)
                input_idx = input_idx[:1+len_box] # bugfix:第一个route给忘了，也就是不包含route往后的东西
                len_route = len_box_route - len_box
                
                gt_supervise = [0,len_box+1] # cls_emb & route # TODO: 后面是考虑多个route了
                pred_supervise = [self.wp_query_pos,self.route_query_pos] # cls_emb & route
                
                # 考虑第2个route:
                if len_route==1:
                    pass
                elif len_route==2:
                    pass
                    # gt_supervise.append(len_box+2)
                    # pred_supervise.append(self.route_query_pos2) # 保证放在第一个route后面
                    # # 没有check这个route_query2上是否蕴含了第二个route的信息
                else:
                    assert False
                    
                for i, obj_idx in enumerate(input_idx):
                    # 这个for循环，似乎无法写成一句话
                    # idx = torch.where(last_layer_batchi_gt==obj_idx)[0] # tuple里面只有一个元素
                    idx = torch.nonzero(last_layer_batchi_gt==obj_idx) # 是50个里面选，进行索引
                    # torch.Size([50]) int -> (x, 1)
                    if len(idx) == 1:
                        gt_i = i+1
                        pred_i = idx.item()+self.extra_query # TODO：如果route数量不固定，需要动态的
                        # assert gt_i >= 0 and gt_i < len_gtmap # 这个包含了pad所以比较宽泛
                        try:
                            assert gt_i >= 0 and gt_i < 1+len_box # 也就是route前面的东西 
                            assert pred_i >=0 and pred_i < len_predmap
                        except:
                            import pdb;pdb.set_trace()
                        gt_supervise.append(gt_i)
                        pred_supervise.append(pred_i)
                    else:
                        assert len(idx) == 0
                
                gt_supervise = gt_attnmap.new_tensor(np.array(gt_supervise)).to(torch.long)
                gt_attnmap_filter = gt_attnmap[:,:,gt_supervise][:,:,:,gt_supervise] # TODO
                gt_attnmap_filter_sum1 = gt_attnmap_filter.sum(-1)
                gt_attnmap_filter_sum1 += 0.00001
                gt_attnmap_filter = (gt_attnmap_filter / gt_attnmap_filter_sum1[...,None]) # .clamp(0.00001,1)
                
                pred_supervise = gt_attnmap.new_tensor(np.array(pred_supervise)).to(torch.long)
                pred_attnmap_filter = attnmap[:,:,pred_supervise][:,:,:,pred_supervise]
                pred_attnmap_filter_sum1 = pred_attnmap_filter.sum(-1) # 感觉是因为这个值太小导致的
                pred_attnmap_filter_sum1 += 0.00001
                pred_attnmap_filter = (pred_attnmap_filter / pred_attnmap_filter_sum1[...,None]) # 这样又不是
                
                isnotnan = torch.logical_and(torch.isfinite(pred_attnmap_filter), torch.isfinite(gt_attnmap_filter)) # 应该不会有作用，因为我使用了clamp, torch.Size([1, 8, 15, 15])
                
                # if torch.any(isnotnan==False):
                #     import pdb;pdb.set_trace()
                
                loss_attnmap = F.l1_loss(gt_attnmap_filter[isnotnan], pred_attnmap_filter[isnotnan]) # layer层面
                if torch.isnan(loss_attnmap): ## 遇到了nan，发现是因为isnotnan全是false，学成了这样
                    import pdb;pdb.set_trace()
                    
                loss_attnmap = torch.nan_to_num(loss_attnmap) ## TODO
                # self.loss_update(losses, loss_attnmap, 'attnmap')
                losses.update({'attnmap_loss': loss_attnmap*self.loss_weights.loss_attnmap})
                
        return losses, new_gt_idxs_list_layers
    
    def loss_update(self, losses, losses_x, type_str):
        # TODO: 判断losses的shape
        losses_x_dict = {}
        for i, loss in enumerate(losses_x):
            losses_x_dict[f'd{i}'] = loss
        losses.update(self.loss_add_prefix(losses_x_dict, type_str))
    
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
            elif prefix == 'attnmap':
                loss_factor = 1 / torch.exp(self.attnmap_weight)
                loss_uncertainty = 0.5 * self.loss_weights.loss_attnmap
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
            elif prefix == 'attnmap':
                loss_factor = self.loss_weights.loss_attnmap
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
        preds_dicts = self.bbox_coder.decode(outs, no_filter=no_filter)
        # [dict_keys(['bboxes', 'scores', 'labels']),...]
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes'] # torch.Size([300, 9])
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, 9)
            scores = preds['scores'] # torch.Size([300])
            labels = preds['labels'] # torch.Size([300])
            wp_attn = preds['wp_attn'] # torch.Size([300])
            wp_attn = preds['wp_attn'] # torch.Size([300])
            matched_idxs = preds['matched_idxs']
            ret_list.append([bboxes, scores, labels, wp_attn, matched_idxs])
        
        if for_aux_outs:
            return [dict(
                boxes_3d=bboxes,
                scores_3d=scores,
                labels_3d=labels,
                wp_attn=wp_attn,
                matched_idxs=matched_idxs,
                # attrs_3d=outs['all_wp_preds'][-1][i] # 每个bs的最后一层
            ) for i, (bboxes, scores, labels, wp_attn, matched_idxs) in enumerate(ret_list)]
        
        bbox_results = []
        for i, (bboxes, scores, labels, wp_attn, matched_idxs) in enumerate(ret_list):
            attrs = None
            refine_wp = None
            route_wp = None
            iscollide = None
            fut_boxes = None
            attnmap = None
            if outs['all_wp_preds'] is not None:
                attrs=outs['all_wp_preds'][-1][i].cpu() # 最后一层的第i个batch
            if outs['refine_wp'] is not None:
                refine_wp=outs['refine_wp'][-1][i].cpu() # 最后一层的第i个batch
            if outs['route_wp'] is not None:
                route_wp=outs['route_wp'][-1][i].cpu()
            if outs['iscollide'] is not None:
                iscollide=outs['iscollide'][i].cpu() # bool
            if outs['fut_boxes'] is not None:
                fut_boxes=outs['fut_boxes'][i].cpu()
            if outs['attnmap'] is not None:
                attnmap=outs['attnmap'][-1][i].cpu()
                attnmap=torch.cat([t for t in attnmap], axis=1)
            pts_bbox = dict(
                boxes_3d=bboxes.to('cpu'),
                scores_3d=scores.cpu(),
                labels_3d=labels.cpu(),
                attrs_3d=attrs,
                matched_idxs=matched_idxs,
                refine_wp=refine_wp,
                route_wp=route_wp,
                iscollide=iscollide, 
                fut_boxes=fut_boxes,
                attnmap=attnmap,
                wp_attn=wp_attn,
                img_metas=img_metas[i] # 这个bs的img_metas
            )
            bbox_results.append(pts_bbox)
        return bbox_results

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