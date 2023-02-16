import logging
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
# from projects.mmdet3d_plugin import *
from mmdet.models.utils.transformer import DetrTransformerEncoder # 没有的话找不到这个class
import pickle
import os.path as osp
import mmcv

from transformers import (
    AutoConfig,
    AutoModel,
)

logger = logging.getLogger(__name__)
from mmcv.cnn import Linear, bias_init_with_prob

from mmcv.cnn import xavier_init, constant_init
def init_weights(self):
    """Initialize the transformer weights."""
    # bias_init = bias_init_with_prob(0.01)
    # nn.init.constant_(p, bias_init)
    for n, p in self.named_parameters():
        # print(n)
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else: # bias
            # torch.Size([512])
            # nn.init.xavier_uniform_(p) 
            if 'weight' in n:
                nn.init.constant_(p, 1)
            elif 'bias' in n:
                nn.init.constant_(p, 0)
            else:
                import pdb;pdb.set_trace()
            
    # for n, module in self.named_modules(): # 递归迭代的
    #     # if isinstance(module, (nn.Linear, nn.Embedding)):
    #     #     module.weight.data.normal_(mean=0.0, std=0.02)
    #     #     if isinstance(module, nn.Linear) and module.bias is not None:
    #     #         module.bias.data.zero_()
    #     if isinstance(module, nn.LayerNorm):
    #         # print(2)
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    #     else:
    #         init_weights(module)
        # BertLayer
        
    # xavier_init(self.reference_points, distribution='uniform', bias=0.)
def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        print(1)
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        print(2)
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    else:
        print(type(module))
        
class HFLM(nn.Module):
    def __init__(self, config_net, config_all):
        super().__init__()
        self.config_all = config_all
        self.config_net = config_net

        self.object_types = 2 + 1  # vehicles, route +1 for padding and wp embedding
        self.num_attributes = 6  # x,y,yaw,speed/id, extent x, extent y

        precisions = [
            self.config_all.model.pre_training.get("precision_pos", 4),
            self.config_all.model.pre_training.get("precision_pos", 4),
            self.config_all.model.pre_training.get("precision_angle", 4),
            self.config_all.model.pre_training.get("precision_speed", 4),
            self.config_all.model.pre_training.get("precision_pos", 4),
            self.config_all.model.pre_training.get("precision_pos", 4),
        ]

        self.vocab_size = [2**i for i in precisions]

        # model
        config = AutoConfig.from_pretrained(
            self.config_net.hf_checkpoint
        )  # load config from hugging face model
        # prajjwal1/bert-medium
        # print(config)
        # import pdb;pdb.set_trace()
        n_embd = config.hidden_size
        
        from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
        encoder=dict(
            type='DetrTransformerEncoder',
            # num_layers=6,
            num_layers=8,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        # embed_dims=256,
                        embed_dims=n_embd, # 512
                        num_heads=8,
                        dropout=0.1)
                ],
                ffn_cfgs=dict(
                     type='FFN',
                    #  embed_dims=256,
                     embed_dims=n_embd,
                    #  feedforward_channels=1024,
                     feedforward_channels=2048,
                     num_fcs=2,
                    #  ffn_drop=0.,
                     ffn_drop=0.1,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                # feedforward_channels=2048,
                # ffn_dropout=0.1, # dep
                operation_order=('self_attn', 'norm', 'ffn', 'norm')))
        
        if 'tr_type' not in self.config_all or self.config_all.tr_type==None:
            # config
            # num_attention_heads=4 # 必须是ndim=512的因子，这样才可以进行划分
            # config.num_attention_heads = num_attention_heads
            self.model = AutoModel.from_config(config=config)
            # init_weights(self.model)
        elif self.config_all.tr_type=='mm':
            self.model = build_transformer_layer_sequence(encoder)
        elif self.config_all.tr_type=='yhq':
            from training.PlanT.detr_encoder import DETREncoder
            self.model = DETREncoder()
        else:
            assert False

        # sequence padding for batching
        self.cls_emb = nn.Parameter(
            torch.randn(1, self.num_attributes + 1)
        )  # +1 because at this step we still have the type indicator
        self.eos_emb = nn.Parameter(
            torch.randn(1, self.num_attributes + 1)
        )  # unnecessary TODO: remove

        # token embedding
        self.tok_emb = nn.Linear(self.num_attributes, n_embd)
        # object type embedding
        self.obj_token = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, self.num_attributes))
                for _ in range(self.object_types)
            ]
        )
        self.obj_emb = nn.ModuleList(
            [nn.Linear(self.num_attributes, n_embd) for _ in range(self.object_types)]
        )
        self.drop = nn.Dropout(config_net.embd_pdrop)

        # decoder head forecasting
        if (
            self.config_all.model.pre_training.get("pretraining", "none") == "forecast"
            or self.config_all.model.training.get("pretraining_path", "none") != "none"
        ):
            # one head for each attribute type -> we have different precision per attribute
            self.heads = nn.ModuleList(
                [
                    nn.Linear(n_embd, self.vocab_size[i])
                    for i in range(self.num_attributes)
                ]
            )

        # wp (CLS) decoding
        self.wp_head = nn.Linear(n_embd, 64)
        self.wp_decoder = nn.GRUCell(input_size=4, hidden_size=65)
        self.wp_relu = nn.ReLU()
        self.wp_output = nn.Linear(65, 2)

        # PID controller
        self.turn_controller = PIDController(K_P=0.9, K_I=0.75, K_D=0.3, n=20)
        self.speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=20)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        # decay = set()
        # no_decay = set()
        # whitelist_weight_modules = torch.nn.Linear
        # blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        # for mn, m in self.named_modules():
        #     for pn, p in m.named_parameters():
        #         fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

        #         if pn.endswith("bias"):
        #             # all biases will not be decayed
        #             no_decay.add(fpn)
        #         elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
        #             # weights of whitelist modules will be weight decayed
        #             decay.add(fpn)
        #         elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
        #             # weights of blacklist modules will NOT be weight decayed
        #             no_decay.add(fpn)
        #         elif pn.endswith("_ih") or pn.endswith("_hh"):
        #             # all recurrent weights will not be decayed
        #             no_decay.add(fpn)
        #         elif pn.endswith("_emb") or "_token" in pn:
        #             no_decay.add(fpn)

        # # validate that we considered every parameter
        # param_dict = {pn: p for pn, p in self.named_parameters()}
        # inter_params = decay & no_decay
        # union_params = decay | no_decay
        # assert (
        #     len(inter_params) == 0
        # ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        # assert (
        #     len(param_dict.keys() - union_params) == 0
        # ), "parameters %s were not separated into either decay/no_decay set!" % (
        #     str(param_dict.keys() - union_params),
        # )

        # # create the pytorch optimizer object
        # optim_groups = [
        #     {
        #         "params": [param_dict[pn] for pn in sorted(list(decay))],
        #         "weight_decay": train_config.weight_decay,
        #     },
        #     {
        #         "params": [param_dict[pn] for pn in sorted(list(no_decay))],
        #         "weight_decay": 0.0,
        #     },
        # ]
        # optimizer = torch.optim.AdamW(
        #     optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        # )
        # optim_groups = [
        #     {
        #         "params": self.parameters(),
        #         "weight_decay": train_config.weight_decay,
        #     }]
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer


    def forward(self, idx, target=None, target_point=None, light_hazard=None, input_idx_batch=None, output_idx_batch=None, output_disappear_batch=None, output_label_path_batch=None):

        if self.config_all.model.pre_training.get("pretraining", "none") == "none":
            assert (
                target_point is not None
            ), "target_point must be provided for wp output"
            assert (
                light_hazard is not None
            ), "light_hazard must be provided for wp output"

        # create batch of same size as input
        x_batched = torch.cat(idx, dim=0) # torch.Size([554, 8])
        # [torch.Size([18, 8]), torch.Size([25, 8])]...
        # 8个字段分别表示：
        input_batch = self.pad_sequence_batch(x_batched) # torch.Size([32, 25, 7])
        input_batch_type = input_batch[:, :, 0]  # torch.Size([32, 25]): car or map
        # 其中第一个cls_emb，后面是1/2，再后面是eso_emb，最后是0是pad
        input_batch_data = input_batch[:, :, 1:]
        bs = len(input_batch)

        # create same for output in case of multitask training to use this as ground truth
        if target is not None:
            y_batched = torch.cat(target, dim=0)
            output_batch = self.pad_sequence_batch(y_batched)
            output_batch_type = output_batch[:, :, 0]  # car or map
            output_batch_data = output_batch[:, :, 1:]

        # create masks by object type
        car_mask = (input_batch_type == 1).unsqueeze(-1)
        route_mask = (input_batch_type == 2).unsqueeze(-1)
        other_mask = torch.logical_and(route_mask.logical_not(), car_mask.logical_not())
        masks = [car_mask, route_mask, other_mask]

        # get size of input
        (B, O, A) = (input_batch_data.shape)  # batch size, number of objects, number of attributes

        # embed tokens object wise (one object -> one token embedding)
        input_batch_data = rearrange(
            input_batch_data, "b objects attributes -> (b objects) attributes"
        )
        embedding = self.tok_emb(input_batch_data)
        embedding = rearrange(embedding, "(b o) features -> b o features", b=B, o=O)

        # create object type embedding
        obj_embeddings = [
            self.obj_emb[i](self.obj_token[i]) for i in range(self.object_types)
        ]  # list of a tensors of size 1 x features

        # add object type embedding to embedding (mask needed to only add to the correct tokens)
        embedding = [
            (embedding + obj_embeddings[i]) * masks[i] for i in range(self.object_types)
        ]
        embedding = torch.sum(torch.stack(embedding, dim=1), dim=1)

        # embedding dropout
        x = self.drop(embedding)

        if 'tr_type' not in self.config_all or self.config_all.tr_type==None:
            # Transformer Encoder; use embedding for hugging face model and get output states and attention map
            output = self.model(**{"inputs_embeds": embedding}, output_attentions=True)
            x, attn_map = output.last_hidden_state, output.attentions
            # x: torch.Size([32, 25, 512])
            # attn_map: (32,head,25,25)*layers=8
            if input_idx_batch is not None:
                attn_map = torch.stack(attn_map, 1) # bs, layer, head, 25, 25
                for batch_id in range(bs):
                    info = dict(
                        attn_map=attn_map[batch_id].cpu().detach().numpy(), # layer, head, 25, 25
                        input_idx=input_idx_batch[batch_id].cpu().detach().numpy(), 
                        output_idx=output_idx_batch[batch_id].cpu().detach().numpy(), 
                        output_disappear=output_disappear_batch[batch_id].cpu().detach().numpy(),
                        output_label_path=output_label_path_batch[batch_id].astype(str)
                    )
                    dirname = '/'.join(info['output_label_path'].split('/')[:-2])+'/attnmap'
                    mmcv.mkdir_or_exist(dirname)
                    basename = osp.basename(info['output_label_path']).replace('json','pkl')
                    path = dirname + '/' + basename
                    # import pdb;pdb.set_trace()
                    with open(path,'wb') as f: 
                        pickle.dump(info,f,protocol=pickle.HIGHEST_PROTOCOL)
                    
        elif self.config_all.tr_type=='mm':
            output = self.model(
                query=embedding,
                key=embedding,
                value=embedding,
            )# 没有pos
            x = output
            attn_map = None
        elif self.config_all.tr_type=='yhq':
            x = self.model(embedding)
            attn_map = None
        else:
            assert False

        # forecasting encoding
        if (
            self.config_all.model.pre_training.get("pretraining", "none") == "forecast"
            or self.config_all.model.training.get("pretraining_path", "none") != "none"
        ):
            car_mask_output = (output_batch_type == 1).unsqueeze(-1)
            non_car_mask_output = (output_batch_type != 1).unsqueeze(-1)
            # size: list of self.num_attributes tensors of size B x O x vocab_size (vocab_size differs for each attribute)
            # we do forecasting only for vehicles
            logits = [
                self.heads[i](x) * car_mask_output - 999 * non_car_mask_output
                for i in range(self.num_attributes)
            ]
            logits = [
                rearrange(logit, "b o vocab_size -> (b o) vocab_size")
                for logit in logits
            ]

            # get target (GT) in same shape as logits
            targets = [
                output_batch_data[:, :, i].unsqueeze(-1) * car_mask_output - 999 * non_car_mask_output
                for i in range(self.num_attributes)
            ]
            targets = [
                rearrange(target, "b o vocab_size -> (b o) vocab_size").long()
                for target in targets
            ]

            # if we do pre-training (not multitask) we don't need wp for pre-trining step so we can return here
            if (
                self.config_all.model.pre_training.get("pretraining", "none") == "forecast"
                and self.config_all.model.pre_training.get("multitask", False) == False
            ):
                return logits, targets
        else:
            logits = None

        # get waypoint predictions
        z = self.wp_head(x[:, 0, :])
        # add traffic ligth flag
        z = torch.cat((z, light_hazard), 1)

        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype)
        x = x.type_as(z)

        # autoregressive generation of output waypoints
        for _ in range(self.config_all.model.training.pred_len):
            x_in = torch.cat([x, target_point], dim=1)
            z = self.wp_decoder(x_in, z)
            dx = self.wp_output(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)

        # pred the wapoints in the vehicle coordinate and we convert it to lidar coordinate here because the GT waypoints is in lidar coordinate
        pred_wp[:, :, 0] = pred_wp[:, :, 0] - 1.3

        if (
            self.config_all.model.pre_training.get("pretraining", "none") == "forecast"
            and self.config_all.model.pre_training.get("multitask", False) == True
        ):
            return logits, targets, pred_wp, attn_map
        else:
            return logits, targets, pred_wp, attn_map


    def pad_sequence_batch(self, x_batched):
        """
        Pads a batch of sequences to the longest sequence in the batch.
        """
        # split input into components
        x_batch_ids = x_batched[:, 0]

        x_tokens = x_batched[:, 1:]

        B = int(x_batch_ids[-1].item()) + 1
        input_batch = []
        for batch_id in range(B):
            # get the batch of elements
            x_batch_id_mask = x_batch_ids == batch_id

            # get the batch of types
            x_tokens_batch = x_tokens[x_batch_id_mask]

            x_seq = torch.cat([self.cls_emb, x_tokens_batch, self.eos_emb], dim=0)

            input_batch.append(x_seq)

        padded = torch.swapaxes(pad_sequence(input_batch), 0, 1)
        input_batch = padded[:B]

        return input_batch


    def control_pid(self, waypoints, velocity, is_stuck=False):
        """Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            velocity (tensor): speedometer input
        """
        assert waypoints.size(0) == 1
        waypoints = waypoints[0].data.cpu().numpy()
        # when training we transform the waypoints to lidar coordinate, so we need to change is back when control
        waypoints[:, 0] += 1.3

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
