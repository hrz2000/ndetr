from re import T
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
import mmcv
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.detr3d_attn_ca import Detr3DCrossAtten

@TRANSFORMER.register_module()
class Detr3DTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 two_stage_num_proposals=300,
                 decoder=None,
                use_wp_query = True,
                use_bev_query = True,
                use_route_query = True,
                use_route_query2 = False,
                route_num_attributes = 6,
                use_type_emb = True,
                wp_refine = None,
                temporal = None,
                route_mask = 0.0,
                no_route = False,
                feat_init = False,
                feat_mlp = False,
                use_intend = False,
                 **kwargs):
        
        super(Detr3DTransformer, self).__init__(**kwargs)
        self.use_intend = use_intend
        self.feat_mlp = feat_mlp
        self.feat_init = feat_init
        self.route_mask = route_mask
        self.no_route = no_route
        
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        
        self.use_wp_query = use_wp_query
        self.use_bev_query = use_bev_query
        self.use_route_query = use_route_query
        self.use_route_query2 = use_route_query2 # TODO
        self.extra_query = 1
        if self.use_bev_query:
            self.extra_query += 1
        if self.use_route_query:
            self.extra_query += 1
        
        self.route_num_attributes = route_num_attributes
        self.use_type_emb = use_type_emb
        self.wp_refine = wp_refine
        self.temporal = temporal
        
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        bev_size = 300
        self.bev_down_size = bev_size // 16
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.obj_type_emb = nn.Embedding(1, self.embed_dims)
        if self.use_bev_query:
            self.bev_emb = nn.Linear(2*self.bev_down_size*self.bev_down_size, self.embed_dims*2)
            if self.use_type_emb:
                self.bev_type_emb = nn.Embedding(1, self.embed_dims)
        if self.use_route_query: # ????????????????????????route_batch?????????
            self.route_emb = nn.Linear(self.route_num_attributes, self.embed_dims*2) # ????????????????????????
            if self.use_type_emb:
                self.route_type_emb = nn.Embedding(1, self.embed_dims)
        if self.use_wp_query:
            if self.feat_init:
                if self.feat_mlp:
                    self.wp_emb_proj = nn.Sequential( # bs, 1000
                                nn.Linear(1000, 512),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, 512),
                                nn.Dropout2d(p=0.5),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, self.embed_dims*2),
                            )
                else:
                    self.wp_emb_proj = nn.Linear(1000, self.embed_dims*2)
            else:
                self.wp_emb = nn.Embedding(1, self.embed_dims*2)
            if self.use_intend:
                self.mea_emb_proj = nn.Linear(3, self.embed_dims*2) # cmd, tp
            if self.use_type_emb:
                self.wp_type_emb = nn.Embedding(1, self.embed_dims)
        self.wp_query_pos = 0
        self.route_query_pos = 1 # ???????????????wp_query
        self.route_query_pos2 = 2 # ???????????????wp_query, batch????????????????????????????????????
        
        if self.wp_refine: # TODO,
            self.reference_points2 = nn.Linear(self.embed_dims, 8)

        if self.temporal == 'bevformer':
            # self.temporal_attn = nn.MultiheadAttention(ndim, nhead, dropout=dropout)
            args = dict(
                dropout = 0., # TODO
                hidden_dim = self.embed_dims,
                dim_in = self.embed_dims,
                update_query_pos = True
            )
            dropout = args['dropout']
            dim_in = args['dim_in']
            hidden_dim = args['hidden_dim']

            self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
            self.linear1 = nn.Linear(dim_in, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(hidden_dim, dim_in)

            if args['update_query_pos']:
                self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
                self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
                self.dropout_pos1 = nn.Dropout(dropout)
                self.dropout_pos2 = nn.Dropout(dropout)
                self.norm_pos = nn.LayerNorm(dim_in)

            self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
            self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_feat1 = nn.Dropout(dropout)
            self.dropout_feat2 = nn.Dropout(dropout)
            self.norm_feat = nn.LayerNorm(dim_in)

            self.norm1 = nn.LayerNorm(dim_in)
            self.norm2 = nn.LayerNorm(dim_in)

            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.activation = F.relu
        else:
            pass

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(m, Detr3DCrossAtten):
                m.init_weight()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
    
    def concat_other_query(self, query, query_pos, img_metas, bs, flatten_feat, tp_batch=None, cmd_batch=None):
        if self.use_type_emb:
            query = query + self.obj_type_emb.weight

        if self.use_bev_query:
            if not self.no_route:
                bev_batch = np.stack([t['hdmap'] for t in img_metas])
                bev_batch = query.new_tensor(bev_batch)
                bev_batch = F.interpolate(
                    bev_batch, size=(self.bev_down_size, self.bev_down_size), mode='bilinear', align_corners=False).flatten(1) 
                bev_emb = self.bev_emb(bev_batch)  # 2*18*18->256d
            else:
                bs = len(img_metas)
                bev_emb = query.new_tensor(np.random.randn(bs, 512))  # 2*18*18->256d
                
            bev_pos, bev_emb = torch.split(bev_emb, self.embed_dims , dim=-1)
            if self.use_type_emb:
                bev_emb = bev_emb + self.bev_type_emb.weight

            query_pos = torch.cat([bev_pos.unsqueeze(1), query_pos],dim=1)
            query = torch.cat([bev_emb.unsqueeze(1), query],dim=1)
        
        if self.use_route_query: # ??????route query????????????
            if not self.no_route:
                if np.random.rand() < self.route_mask:
                    bs = len(img_metas)
                    route_emb = query.new_tensor(np.random.randn(bs, 512))  # 2*18*18->256d
                else:
                    route_batch = np.stack([t['plan']['route'][0] for t in img_metas]) # bs,6
                    route_batch = query.new_tensor(route_batch)
                    route_emb = self.route_emb(route_batch)
            else:
                bs = len(img_metas)
                route_emb = query.new_tensor(np.random.randn(bs, 512))  # 2*18*18->256d
                
            route_pos, route_emb = torch.split(route_emb, self.embed_dims , dim=-1)
            if self.use_type_emb:
                route_emb = route_emb + self.route_type_emb.weight
            query_pos = torch.cat([route_pos.unsqueeze(1), query_pos],dim=1)
            query = torch.cat([route_emb.unsqueeze(1), query],dim=1)

        if self.use_wp_query:
            if self.feat_init:
                wp_emb = self.wp_emb_proj(flatten_feat)
            else:
                wp_emb = self.wp_emb.weight
                wp_emb = wp_emb.expand(bs, -1)
            if self.use_intend:
                means_emb = self.mea_emb_proj(torch.cat([cmd_batch, tp_batch], dim=-1))
                wp_emb = wp_emb + means_emb
            wp_pos, wp_emb = torch.split(wp_emb, self.embed_dims, dim=-1)
            if self.use_type_emb:
                wp_emb = wp_emb + self.wp_type_emb.weight
            query_pos = torch.cat([wp_pos.unsqueeze(1), query_pos],dim=1)
            query = torch.cat([wp_emb.unsqueeze(1), query],dim=1)

        return query, query_pos

    def qim_temporal_attn(self, query, query_pos):
        q = k = query_pos + query
        
        tgt = query
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # ?????????????????????????????????

        if self.update_query_pos:
            # ffn: linear_pos2
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)

        query2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query = query + self.dropout_feat2(query2)
        query = self.norm_feat(query) # ???????????????ffn??????????????????
        return query, query_pos

    def forward(self,
                mlvl_feats,
                query_embed,
                reg_branches=None,
                wp_branches=None,
                img_metas=None,
                prev_bev=None,
                tp_batch=None,
                light_batch=None,
                cmd_batch=None,
                route_batch=None,
                batch_npred_bbox=None,
                flatten_feat=None,
                **kwargs):
        assert query_embed is not None
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(query_embed, self.embed_dims , dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        query, query_pos = self.concat_other_query(query, query_pos, img_metas, bs, flatten_feat=flatten_feat, tp_batch=tp_batch, cmd_batch=cmd_batch) # torch.Size([bs=42, numq=53, 256])
        # 53=1wp+1route+1hdmap
        
        if prev_bev is not None:
            assert prev_bev.shape == query.shape
            bs, numq, ndim = prev_bev.shape
            mask = [0,*range(3,numq)] # ??????1route?????????2hdmap
            query[:,mask], query_pos[:,mask] = self.qim_temporal_attn(prev_bev[:,mask], query_pos[:,mask]) # ????????????emb?????????????????????emb???
            # query[:,mask], attn_weight_ = self.temporal_attn(query=q[:,mask], key=k[:,mask], value=prev_bev[:,mask], key_padding_mask=None) 
            # torch.Size([bs=1, 53, 256]), torch.Size([53, 1, 1]) weight???query?????????key?????????
            # ?????????qim?????????tracked???query????????????
            
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid() # torch.Size([42, 53, 3])
        init_reference_out = reference_points
        
        if self.wp_refine:
            reference_points2 = self.reference_points2(query_pos[:, self.wp_query_pos]).reshape(bs, 4, 2)
            # reference_points2 = torch.cat([reference_points2, reference_points2.new_zeros((bs,4,1))], dim=-1)
            reference_points2 = reference_points2.sigmoid() # torch.Size([42, 4, 3])
            init_reference_out2 = reference_points2
        else:
            reference_points2 = None
            init_reference_out2 = None
            
        # decoder
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references, inter_references2, all_attnmap = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reference_points2=reference_points2,
            reg_branches=reg_branches,
            wp_branches=wp_branches,
            extra_query=self.extra_query,
            wp_query_pos=self.wp_query_pos,
            img_metas=img_metas,
            tp_batch=tp_batch,
            light_batch=light_batch,
            cmd_batch=cmd_batch,
            route_batch=route_batch,
            batch_npred_bbox=batch_npred_bbox,
            **kwargs)
        # [torch.Size([6, 53, 2, 256]), torch.Size([6, 2, 53, 3]), torch.Size([6, 2, 4, 2])]

        inter_references_out = inter_references
        inter_references_out2 = inter_references2
        return inter_states, init_reference_out, inter_references_out, init_reference_out2, inter_references_out2, all_attnmap
