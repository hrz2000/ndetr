_base_ = ['../../../../mmdetection3d/configs/_base_/datasets/nus-3d.py']
from projects.configs.detr3d.new.common import *

use_fv=False

batch=32
workers=4
lr=2e-4
num_query=50

wp_refine=None # gru, linear, None
wp_refine_input_last=False

gru_use_box=0
velo_update=False

penalty_args = dict(
    use_route_penalty=False,
    use_collide_penalty=False,
    use_comfort_penalty=False,
    use_progress_penalty=False,
)

enable_uncertainty_loss_weight=False

temporal=None 
# temporal='bevformer' # bevformer,mutr3d,gruinfer
find_unused_parameters=True # no_grad的时候，需要

#################

model = dict(
    type='Detr3D',
    use_grid_mask=True,
    temporal=temporal,
    use_fv=use_fv,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe'),
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='AttnHead',
        num_query=num_query,
        num_classes=len(class_names),
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        use_proj=False,
        use_cmd=True,
        gru_use_box=gru_use_box,
        penalty_args=penalty_args,
        wp_refine = wp_refine,
        wp_refine_input_last = wp_refine_input_last,
        velo_update = velo_update,
        enable_uncertainty_loss_weight = enable_uncertainty_loss_weight,
        transformer=dict(
            type='Detr3DTransformer',
            use_wp_query = True,
            use_bev_query = True,
            use_route_query = True,
            route_num_attributes = 6,
            use_type_emb = True,
            wp_refine = wp_refine,
            temporal = temporal,
            decoder=dict(
                type='Detr3DTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                wp_refine=wp_refine,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='Detr3DCrossAtten',
                            pc_range=point_cloud_range,
                            wp_refine=wp_refine,
                            num_points=1,
                            num_cams=1,
                            embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=num_query,
            voxel_size=voxel_size,
            score_threshold=0.3,
            num_classes=len(class_names)), 
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range))))

# dataset_type = 'NuScenesDataset'
dataset_type = 'CustomNuScenesDataset'

file_client_args = dict(backend='disk')

# db_sampler = dict(
#     data_root=data_root,
#     info_path=data_root + 'carla_train.pkl',
#     rate=1.0,
#     prepare=dict(
#         filter_by_difficulty=[-1],
#         filter_by_min_points=dict(
#             car=5,
#             truck=5,
#             bus=5,
#             trailer=5,
#             construction_vehicle=5,
#             traffic_cone=5,
#             barrier=5,
#             motorcycle=5,
#             bicycle=5,
#             pedestrian=5)),
#     classes=class_names,
#     sample_groups=dict(
#         car=2,
#         truck=3,
#         construction_vehicle=7,
#         bus=4,
#         trailer=6,
#         barrier=2,
#         motorcycle=6,
#         bicycle=6,
#         pedestrian=2,
#         traffic_cone=2),
#     points_loader=dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=5,
#         use_dim=[0, 1, 2, 3, 4],
#         file_client_args=file_client_args))

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles2', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3Dx', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img']) # 'test'
]
test_pipeline = train_pipeline
# test_pipeline = [
#     dict(type='LoadMultiViewImageFromFiles', to_float32=True),
#     dict(type='NormalizeMultiviewImage', **img_norm_cfg),
#     dict(type='PadMultiViewImage', size_divisor=32),
#     dict(
#         type='MultiScaleFlipAug3D',
#         img_scale=(1333, 800),
#         pts_scale_ratio=1,
#         flip=False,
#         transforms=[
#             dict(
#                 type='DefaultFormatBundle3D',
#                 class_names=class_names,
#                 with_label=False),
#             dict(type='Collect3Dx', keys=['img'])
#         ])
# ]
common_data = dict(
        type=dataset_type,
        with_velocity=True,
        classes=class_names,
        modality=input_modality,
        use_valid_flag=True,
        box_type_3d='LiDAR',
        img_shape=img_shape,
        is_carla=is_carla,
        clss_range=clss_range,
        debug=debug,
        vis_dir=vis_dir,
        temporal=temporal,
)
train_data=dict(
        **common_data,
        data_root=data_root,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        test_mode=False,
        vis=False, 
)
test_data=dict(
        **common_data,
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        test_mode=False,
        vis=False
)
# train_data = test_data

data = dict(
    samples_per_gpu=batch,
    workers_per_gpu=workers,
    train=train_data,
    val=test_data,
    test=test_data)

optimizer = dict(
    type='AdamW', 
    lr=lr,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 15
evaluation = dict(interval=3, pipeline=test_pipeline, save_best="wp", less_keys=['wp'])

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# load_from='pretrain/route.pth'
# load_from='pretrain/detr3d_101.pth'
# resume_from = 'work_dirs/ndetr_hdmap/epoch_15.pth'
# find_unused_parameters=True
