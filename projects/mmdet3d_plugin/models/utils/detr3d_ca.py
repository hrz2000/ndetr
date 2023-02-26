
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.runner.base_module import BaseModule
from .tools import inverse_sigmoid
import warnings
from projects.mmdet3d_plugin.datasets.vis_tools import wp23d
import math

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x = tensor_list.tensors
        mask = x.new_zeros(x.shape).to(torch.bool)
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class CrossAttn(BaseModule):
    def __init__(self,
                 **kwargs):

        super(CrossAttn, self).__init__()
        args = dict(
            dropout = 0.1, # TODO
            hidden_dim = 256,
            dim_in = 256,
            update_query_pos = True
        )
        dropout = args['dropout']
        dim_in = args['dim_in']
        hidden_dim = args['hidden_dim']
        update_query_pos = args['update_query_pos']
        self.update_query_pos = update_query_pos
        
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
        
    def forward(self,query,query_pos,key,key_pos,use_fv_sinpe=True):
        '''
        key: [torch.Size([2, 1, 256, 32, 116]), torch.Size([2, 1, 256, 16, 58]), torch.Size([2, 1, 256, 8, 29]), torch.Size([2, 1, 256, 4, 15])]
        query: torch.Size([1, 2, 256])
        '''
        q = query_pos + query
        tgt = query
        
        if use_fv_sinpe:
            pe = PositionEmbeddingSine(num_pos_feats=256/2, normalize=True)
            h, w = key.shape[-2:]
            key_pos = pe(key[0, 0:1]) # 取bs第1个, channels也不用传入太多
            key_pos = key_pos.flatten(2).permute(2,0,1) # c=1,256,h*w
            
            # key: bs, channels, h, w
            key = key.flatten(2).permute(2,0,1)
            k = key + key_pos
            v = key
        else:
            k = key
            v = key
            
        tgt2, attn_map = self.self_attn(q, k, v) # 8头返回平均情况
        attn_map = attn_map.reshape(h, w)
        # tgt2: torch.Size([1, 2, 256])
        # attn_map: torch.Size([2, 1, 3712])
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            # ffn: linear_pos2
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)

        query2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query = query + self.dropout_feat2(query2)
        query = self.norm_feat(query)
        
        return query, query_pos

@ATTENTION.register_module()
class Detr3DCrossAtten(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 wp_refine=False,
                 batch_first=False,
                 wp_global_attn=False):
        super(Detr3DCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.wp_refine = wp_refine
        self.wp_global_attn = wp_global_attn

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams*num_levels*num_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)
      
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        
        if self.wp_global_attn:
            self.cross_attn = CrossAttn()
        elif self.wp_refine: # TODO: 应该改一下逻辑
            num_points = 4
            self.attention_weights2 = nn.Linear(embed_dims,
                                            num_cams*num_levels*num_points)

            self.output_proj2 = nn.Linear(embed_dims, embed_dims)


        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                reference_points=None,
                reference_points2=None,
                extra_query=0,
                wp_query_pos=0,
                tp_batch=None,
                light_batch=None,
                cmd_batch=None,
                key_padding_mask=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        return self.forward_obj_query(                
                query,
                key,
                value,
                residual=residual,
                query_pos=query_pos,
                reference_points=reference_points,
                reference_points2=reference_points2,
                extra_query=extra_query,
                wp_query_pos=wp_query_pos,
                tp_batch=tp_batch,
                light_batch=light_batch,
                cmd_batch=cmd_batch,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs)

    def forward_obj_query(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                reference_points=None,
                reference_points2=None,
                extra_query=0,
                wp_query_pos=0,
                key_padding_mask=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        num = extra_query
        
        wp_emb = query[wp_query_pos:wp_query_pos+1,:,:] # torch.Size([1, 42, 256])
        wp_emb_pos = query_pos[wp_query_pos:wp_query_pos+1,:,:] # torch.Size([1, 42, 256])
        other_query = query[wp_query_pos+1:num,:,:]
        other_query_pos = query_pos[wp_query_pos+1:num,:,:]

        query = query[num:]                             # torch.Size([50, 42, 256])
        query_pos = query_pos[num:]                     # torch.Size([50, 42, 256])
        reference_points = reference_points[:,num:,:]   # torch.Size([42, 50, 3])

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        else:
            inp_residual = None
        if query_pos is not None:
            query = query + query_pos

        query = query.permute(1, 0, 2) # torch.Size([42, 50, 256])

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        
        reference_points_3d = reference_points.clone()
        output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        # reference_points_3d:torch.Size([42, 50, 3]) -> torch.Size([bs*cam, h=50, w=num_p=1, 2])
        # attention_weights:  torch.Size([42, 1, 50, 1, 1, 4])
        # mask:               torch.Size([42, 1, 50, 1, 1, 1])
        # output:             torch.Size([42, 256, 50, num_cam=1, num_point=1, lvl=4])
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1) # torch.Size([900, 6, 256])
        
        output = self.output_proj(output) # torch.Size([900, 6, 256])
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        output = self.dropout(output) + inp_residual + pos_feat
        
        if self.wp_global_attn:
            fv_feat_lvl1 = value[0]
            fv_feat_cam0 = fv_feat_lvl1[:,0]
            fv_feat = fv_feat_cam0 # bs, c, h, w
            wp_emb, wp_pos = self.cross_attn(query=wp_emb, key=fv_feat, query_pos=wp_emb_pos, key_pos=None, use_fv_sinpe=True)
        elif self.wp_refine:
            # if self.wp_global_attn:
            #     fv_feat_lvl1 = value[0]
            #     fv_feat_cam0 = fv_feat_lvl1[:,0]
            #     fv_feat = fv_feat_cam0 # bs, c, h, w
            #     wp_emb, wp_pos = self.cross_attn(query=wp_emb, key=fv_feat, query_pos=wp_emb_pos, key_pos=None, use_fv_sinpe=True)
            # else:
            wp_emb = self.attn(wp_emb, None, value, query_pos=wp_emb_pos, reference_points=reference_points2, num_point=4, **kwargs)
        
        output = torch.cat([wp_emb, other_query, output], dim=0)
        return output

    def attn_gloabl(self, query, key, value, query_pos, key_pos):
        pass

    def attn(self,
            query,
            key,
            value,
            residual=None,
            query_pos=None,
            reference_points=None,
            num_point=1,
            extra_query=0,
            tp_batch=None,
            light_batch=None,
            cmd_batch=None,
            key_padding_mask=None,
            spatial_shapes=None,
            level_start_index=None,
            **kwargs):
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        else:
            inp_residual = None
        if query_pos is not None:
            query = query + query_pos

        query = query.permute(1, 0, 2) # torch.Size([42, 1, 256])

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights2(query).view(
            bs, 1, num_query, self.num_cams, num_point, self.num_levels)
        reference_points = wp23d(reference_points, -2.5) # 当前是lidar系的坐标
        # TODO: 前视图来看是没有意义的
        reference_points_3d = reference_points.clone() # torch.Size([42, 4, 3]) 理论上第二个值表示的是num_queries，我们用作num_points
        output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        # reference_points_3d:torch.Size([42, 50, 3]) -> torch.Size([bs*cam, h=50, w=num_p=1, 2])
        # attention_weights:  torch.Size([42, 1, 50/1, 1, 1/4, 4])
        # mask:               torch.Size([42, 1, 50/4, 1, 1, 1])
        # output:             torch.Size([42, 256, 50/4, num_cam=1, num_point=1, lvl=4])
        bs, ndim, num_q, num_cam, num_p, lvl = output.shape
        # mask = mask.reshape(bs, 1, num_point, num_cam, num_q, 1)
        # output = output.reshape(bs, ndim, num_point, num_cam, num_q, lvl)
        mask = mask.permute(0,1,4,3,2,5)
        output = output.permute(0,1,4,3,2,5)
        # 本身就没有多个点这个事情，我们拿num_queries作为点数，这里进行恢复
        
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1) # torch.Size([4, 42, 256])
        
        output = self.output_proj2(output) # torch.Size([900, 6, 256])
        # pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2) # torch.Size([2, 4, 256]) torch.Size([4, 2, 256])
        pos_feat = pos_feat.sum(0, keepdim=True)

        output = self.dropout(output) + inp_residual + pos_feat
        return output


def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1) # torch.Size([6, 1, 4, 4])
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1) # 把参考点重复cam次 torch.Size([6, 对应c的1, 900, xyzd=4, 用来做矩阵乘法的没用的维度1])
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1) # 把矩阵重复query次 B, num_cam, 900, 4, 4
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    # torch.Size([6, 1, 900, 4, 4]) torch.Size([6, 相机重复=1, 900, 4个维度=4, 1])
    # 得到不同相机的img上坐标，还需要除以第三个维度
    # torch.Size([6, 1, 900, 4]) 每个batch的每个相机的900个4d点
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps) # torch.Size([6, 1, 900, 1])
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    # 每个bs都是一样的，每个cam都是一样的，变成在图中hw的比例
    reference_points_cam = (reference_points_cam - 0.5) * 2 # why,变到中心
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0)) # 图外
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    # torch.Size([6, cam=1, 1, 900, 1, 1])->torch.Size([6, channel=1, 900, cam=1, 对应points=1, lvl=1])
    mask = torch.nan_to_num(mask)
    sampled_feats = []
    B, N, C, H, W = mlvl_feats[0].size()
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B*N, C, H, W)
        # torch.Size([6, 256, 32, 116]),这里认为不同cam的hw一样
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 1, 2)
        # torch.Size([6=bs*cam, 900, 1, xy=2])
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl, align_corners=False)
        # torch.Size([6, 256, 900, 1])
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        # torch.Size([6, cam=1, c=256, 900, 1])
        # torch.Size([6, 256, 900, cam=1, 1])
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    # torch.Size([6, 256, 900, cam=1, lvl=4])
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  1, len(mlvl_feats))
    # torch.Size([6, 256, 900, cam=1, 1, lvl=4])
    # mask:torch.Size([6, c=1, 900, cam=1, 对应points=1, lvl=1]),这个只关注于点，不看其他的
    return sampled_feats, mask