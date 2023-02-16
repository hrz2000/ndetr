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
from mmdet3d.core import bbox3d2result

@HEADS.register_module()
class Detr3DHead_gtgru(DETRHead):
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
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 use_proj=False,
                 use_cmd=False,
                 **kwargs):
        self.use_proj = use_proj
        self.use_cmd = use_cmd
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
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
        self.num_cls_fcs = num_cls_fcs - 1
        super(Detr3DHead_gtgru, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

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
        
        self.query_embedding = nn.Embedding(1 + self.num_query,
                                                self.embed_dims * 2)
        n_embd = self.transformer.embed_dims # 256
        query_size = 64
        if self.use_cmd:
            hidden_size = query_size+1+1 # light\command
        else:
            hidden_size = query_size+1
        self.wp_head = nn.Linear(n_embd, query_size)
        # self.wp_decoder = nn.GRUCell(input_size=4, hidden_size=hidden_size)
        num_car = 5
        # input_size = 2+2+num_car*6
        input_size = 2+2+num_car*9
        hidden_size=6+1+1
        self.wp_decoder = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.wp_relu = nn.ReLU()
        self.wp_output = nn.Linear(hidden_size, 2)
        if self.use_proj:
            self.wp_proj = nn.Linear(64+1+2+2, 2)#隐藏维度，红绿灯，tp，当前点

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward_gtgru(self, img_metas, gt_bboxes_3d, gt_labels_3d):
        # 拿到该batch中每个对应的top3 object
        bs = len(img_metas)
        box_batch = []
        for gt_box in gt_bboxes_3d:
            t = gt_box.tensor
            t[0] += 1.3 # lidar->ego
            num_car = 5
            num, dim = t.shape
            if num<num_car:
                n = torch.zeros((num_car,dim))
                n[:num] = t
                t = n
            # 现在t补全了
            # ot = t
            # t = torch.zeros((len(ot),6))
            # t[:,0] -= 1.3
            # t[:,:5] = ot[:,[0,1,5,3,6]]
            # t[:,0] += 1.3
            # t[:,5] = torch.sqrt(ot[:,7]**2+ot[:,8]**2) # 速度的值
            distance = torch.sqrt((t[:,0])**2 + t[:,1]**2)
            gt_box_idx = distance.sort(dim=0)[1][:num_car] # 默认升序，但是没有索引了;torch.sort默认是降序
            gt_box = t[gt_box_idx] # 9,速度是2d的123 456  7 89
            # gt_box.shape # (3,9)
            box_batch.append(gt_box)
        box_batch = torch.stack(box_batch) ## TODO 不知道对不对
        box_batch = box_batch.reshape(bs, -1)
        # route向量、top3 object、light等信息，放到gru里面
        route_batch = torch.tensor(np.stack([t['plan']['route'][0] for t in img_metas])) # bs,6
        tp_batch = torch.tensor(np.stack([m['plan']['tp'] for m in img_metas],axis=0))
        light_batch = torch.tensor(np.stack([m['plan']['light'] for m in img_metas],axis=0).reshape(-1,1))
        cmd_batch = torch.tensor(np.stack([m['plan']['command'] for m in img_metas],axis=0).reshape(-1,1))

        box_batch = box_batch.to(torch.float32).to('cuda')
        route_batch = route_batch.to(torch.float32).to('cuda')
        tp_batch = tp_batch.to(torch.float32).to('cuda')
        light_batch = light_batch.to(torch.float32).to('cuda')
        cmd_batch = cmd_batch.to(torch.float32).to('cuda')

        self.pred_len = 4
        # z = self.wp_head(cls_emb) # bs,256->64
        z = route_batch
        if self.use_cmd:
            z = torch.cat((z, light_batch, cmd_batch), 1) # torch.Size([bs, 7+1+1])
        else:
            z = torch.cat((z, light_batch), 1)
        output_wp = list()
        x = z.new_zeros(size=(z.shape[0], 2))

        for _ in range(self.pred_len): # 4
            x_in = torch.cat([x, tp_batch, box_batch], dim=1) # bs,(2+2+6*3)
            z = self.wp_decoder(x_in, z) # torch.Size([1, 65]) z:torch.Size([2, 8])
            dx = self.wp_output(z) # torch.Size([1, 2])
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1) # bs,4,2
        pred_wp[:, :, 0] = pred_wp[:, :, 0] - 1.3 # 变成了lidar系
        return pred_wp
    
    def loss_gru(self, pred_wp, img_metas):
        gt_wp = np.stack([m['plan']['wp'] for m in img_metas],axis=0)
        gt_wp = pred_wp.new_tensor(gt_wp)
        losses_wp = F.l1_loss(pred_wp, gt_wp) # torch.Size([6, 1, 4, 2])
        loss_dict = {}
        loss_dict['loss_wp'] = losses_wp # 没有TR
        return loss_dict

    def forward(self, mlvl_feats, img_metas):
        query_embeds = self.query_embedding.weight
        
        hs, init_reference, inter_references = self.transformer(
            mlvl_feats,
            query_embeds,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            img_metas=img_metas,
        )
        hs = hs.permute(0, 2, 1, 3) # torch.Size([6, 6, 902, 256])
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        # num = hs.shape[-2] % 10
        outputs_classes = torch.stack(outputs_classes)[:,:,num:,:]
        outputs_coords = torch.stack(outputs_coords)[:,:,num:,:]
        # torch.Size([6, bs=2, 900, cls])

        outs = {
            'all_cls_scores': outputs_classes, # torch.Size([6, 1, 50, 2])
            'all_bbox_preds': outputs_coords,  # torch.Size([6, 1, 50, 10])
            'enc_cls_scores': None,
            'enc_bbox_preds': None, 
        }
        outs['all_wp_preds'] = self.pred_waypoint(hs, img_metas) # torch.Size([6, 1, 4, 2])
        return outs
    
    def pred_waypoint(self, hs, img_metas, img=None):
        self.waypoint_pos = 2
        num_layers = self.transformer.decoder.num_layers
        ndim = self.transformer.decoder.embed_dims
        if isinstance(img, torch.Tensor): # 不传入img的时候不经过这里
            hs = np.zeros((num_layers, len(img_metas), self.num_query + self.transformer.decoder.extra_type, ndim))
            hs = img.new_tensor(hs)
        outputs_wps = []
        tp_batch = np.stack([m['plan']['tp'] for m in img_metas],axis=0)
        light_batch = np.stack([m['plan']['light'] for m in img_metas],axis=0).reshape(-1,1)
        cmd_batch = np.stack([m['plan']['command'] for m in img_metas],axis=0).reshape(-1,1)
        for lvl in range(hs.shape[0]):
            cls_emb = hs[lvl][:,self.waypoint_pos,:]
            outputs_wp = self.pred_waypoint_per_layer(
                cls_emb, cls_emb.new_tensor(tp_batch), cls_emb.new_tensor(light_batch), cls_emb.new_tensor(cmd_batch))
            outputs_wps.append(outputs_wp)
        outputs_wps = torch.stack(outputs_wps)
        return outputs_wps

    def pred_waypoint_per_layer(self, cls_emb, tp_batch, light_batch, cmd_batch=None):
        self.pred_len = 4
        z = self.wp_head(cls_emb) # bs,256->64
        if self.use_cmd:
            z = torch.cat((z, light_batch, cmd_batch), 1) # torch.Size([bs, 64+1+1])
        else:
            z = torch.cat((z, light_batch), 1)
        output_wp = list()
        x = z.new_zeros(size=(z.shape[0], 2))

        for _ in range(self.pred_len): # 4
            if self.use_proj:
                x_in = torch.cat([z, x, tp_batch], dim=1)
                dx = self.wp_proj(x_in)
                x = dx + x
                output_wp.append(x)
            else:
                x_in = torch.cat([x, tp_batch], dim=1) # bs,(2+2)
                z = self.wp_decoder(x_in, z) # torch.Size([1, 65])
                dx = self.wp_output(z) # torch.Size([1, 2])
                x = dx + x
                output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1) # bs,4,2
        pred_wp[:, :, 0] = pred_wp[:, :, 0] - 1.3 # 变成了lidar系
        return pred_wp

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
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0) # 2个类别，但是会有0,1,2
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
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
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range) # torch.Size([900, 9])->torch.Size([900, 10])

        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos) # torch.Size([900, 10])

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             img_metas,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
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

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        # LOSS_WP
        all_wp_preds = preds_dicts['all_wp_preds'] # torch.Size([layer=6, bs=1, 4, 2])
        gt_wp = np.stack([m['plan']['wp'] for m in img_metas],axis=0)
        gt_wp = all_wp_preds.new_tensor(gt_wp)
        gt_wp = gt_wp.unsqueeze(0).repeat(len(all_wp_preds),1,1,1)
        losses_wp = F.l1_loss(all_wp_preds, gt_wp, reduction='none').mean([1,2,3]) # torch.Size([6, 1, 4, 2])
        loss_dict['loss_wp'] = losses_wp[-1]
        for num_dec_layer, loss_cls_i in enumerate(losses_wp[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_wp'] = loss_bbox_i

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, outs, img_metas, rescale=False):
        preds_dicts = self.bbox_coder.decode(outs)
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
            ret_list.append([bboxes, scores, labels])
            
        bbox_results = [
            bbox3d2result(bboxes, scores, labels, wps)
            for (bboxes, scores, labels), wps in zip(ret_list, outs['all_wp_preds'][-1])
            # 拿出最后一层的wp预测结果
        ] 
        return bbox_results
