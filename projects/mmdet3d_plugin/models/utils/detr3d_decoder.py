import torch
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from .tools import inverse_sigmoid

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Detr3DTransformerDecoder(TransformerLayerSequence):
    def __init__(self, *args, return_intermediate=False, wp_refine=False, **kwargs):
        super(Detr3DTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.num_layers = kwargs['num_layers']
        self.wp_refine = wp_refine

    def forward(self,
                query,
                *args,
                reference_points=None,
                reference_points2=None,
                reg_branches=None,
                wp_branches=None,
                extra_query=0,
                wp_query_pos=0,
                tp_batch=None,
                light_batch=None,
                cmd_batch=None,
                route_batch=None,
                batch_npred_bbox=None,
                **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = []
        intermediate_reference_points2 = []
        intermediate_attnmap = []
        for lid, layer in enumerate(self.layers): # 这里的每个layer都是一个self-attn、cross-attn序列
            reference_points_input = reference_points
            reference_points_input2 = reference_points2
            output = layer(
                output,
                *args,
                reference_points=reference_points_input, # 这个也放到了kwargs里面
                reference_points2=reference_points_input2, # 这个也放到了kwargs里面
                extra_query=extra_query,
                wp_query_pos=wp_query_pos,
                **kwargs)
            output, attnmap = output 
            # output:  torch.Size([53, 32, 256])
            # attnmap: torch.Size([32, 53, 53]) / torch.Size([32, 8, 53, 53])
            # import pdb;pdb.set_trace()
            output = output.permute(1, 0, 2)
            # torch.Size([42, 53, 256]) torch.Size([42, 99, 256])
            bs, num_q, ndim = output.shape

            if reg_branches is not None:
                tmp = reg_branches[lid](output) # torch.Size([42, 53, 10])
                
                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points) # torch.Size([42, 53, 3])
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                
                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()
            
            if wp_branches is not None:
                tmp = wp_branches[lid](reference_points2, output[:,wp_query_pos], tp_batch, light_batch, cmd_batch, batch_npred_bbox)
                # output: torch.Size([bs=1, npred=4, xy=2])
                
                new_reference_points2 = torch.zeros_like(reference_points2)
                new_reference_points2[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points2[..., :2])
                new_reference_points2 = new_reference_points2.sigmoid()
                reference_points2 = new_reference_points2.detach()
            
            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_reference_points2.append(reference_points2)
                intermediate_attnmap.append(attnmap)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), None if not self.wp_refine else torch.stack(intermediate_reference_points2), torch.stack(intermediate_attnmap, axis=1) # bs, layer, head, x, x

        return output, reference_points, reference_points2, attnmap
