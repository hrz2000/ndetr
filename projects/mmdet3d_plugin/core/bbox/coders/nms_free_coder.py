import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox


@BBOX_CODERS.register_module()
class NMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, attnmap, new_gt_idxs_list, no_filter):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid() # torch.Size([900, 2])
        scores, indexs = cls_scores.view(-1).topk(max_num) # torch.Size([300]) torch.Size([300])
        labels = indexs % self.num_classes # 类别
        # bbox_index = indexs // self.num_classes # 第i个预测的box的索引
        bbox_index = torch.div(indexs, self.num_classes, rounding_mode='trunc') # 第i个预测的box的索引
        bbox_preds = bbox_preds[bbox_index] # torch.Size([300, 10])
        if attnmap is not None:
            # 8个头需要平均一下
            attnmap = attnmap.mean(0)
            wp_attn = attnmap[0][3:] # torch.Size([300, 10])
            assert wp_attn.shape[-1] == 50
            wp_attn = wp_attn[bbox_index]
        else:
            wp_attn = None
            
        if new_gt_idxs_list is not None:
            new_gt_idxs_list = new_gt_idxs_list[bbox_index] 
            # new_gt_idxs_list是预测的50个box和gt的1对1匹配，可能有的没有匹配上gt
            # 这里的bbox_index是进行了置信度过滤等等操作，是说对50个里面的整体。
            # import pdb;pdb.set_trace()
        else:
            new_gt_idxs_list = None

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range) # torch.Size([300, 9])
        # final_box_preds = bbox_preds
        # TODO

        # pred是xyzxyz yaw1 yaw2 v1 v2
        final_scores = scores 
        final_preds = labels 

        # use score threshold
        if not no_filter:
            if self.score_threshold is not None:
                thresh_mask = final_scores > self.score_threshold
        if self.post_center_range is not None: 
            # tensor([-61.2000, -61.2000, -10.0000,  61.2000,  61.2000,  10.0000])
            # xyz xyz
            if not isinstance(self.post_center_range, torch.Tensor):
                self.post_center_range = torch.tensor(
                    self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if not no_filter:
                if self.score_threshold:
                    mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            if wp_attn is not None:
                wp_attn = wp_attn[mask]
            if new_gt_idxs_list is not None:
                new_gt_idxs_list = new_gt_idxs_list[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
                'wp_attn': wp_attn,
                'matched_idxs': new_gt_idxs_list
            }
            # 所以可能少于300个

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts, no_filter):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1] # torch.Size([bs=1, 900, 2])
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]
        if 'attnmap' in preds_dicts:
            attnmap = preds_dicts['attnmap'][-1]
            # 均拿出最后一层的结果
        else:
            attnmap = [None for i in range(len(all_bbox_preds))]
            
        if 'new_gt_idxs_list_layers' in preds_dicts:
            new_gt_idxs_list = preds_dicts['new_gt_idxs_list_layers'][-1] # 均拿出最后一层的结果
            # import pdb;pdb.set_trace()
        else:
            new_gt_idxs_list = [None for i in range(len(all_bbox_preds))]
        
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i], attnmap[i], new_gt_idxs_list[i], no_filter))
        return predictions_list