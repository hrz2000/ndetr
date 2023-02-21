from projects.configs.detr3d.new.debug import debug
is_carla = True
vis_dir='show_dir'
img_shape=(256, 900)
data_root = None
train_ann_file = './data/carla_train_hdmap_all.pkl'
# train_ann_file = './data/carla_val_hdmap_all.pkl'
val_ann_file = './data/carla_val_hdmap_all.pkl'
# train_ann_file = './data/carla_val_hdmap_all.pkl'
# val_ann_file = './data/carla_val_hdmap_all.pkl'

# simu_params=dict(use_collide=False)
simu_params=dict(use_collide=True)
################################
checkpoint_config = dict(interval=3)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook', 
            init_kwargs=dict(
                project='your-awesome-project',
                resume=False)
                # resume='auto'
            )
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
################################

plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

# For nuScenes we usually do 10-class detection
# class_names = [
#     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
#     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
# ]
class_names = [
    'car', 'pedestrian'
]
clss_range = {'car':50, 'pedestrian':40}

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)