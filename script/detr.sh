# python leaderboard/scripts/run_evaluation.py experiments=datagen eval.route_rel=leaderboard/data/training/routes/ll/debug.xml eval.scenarios_rel=leaderboard/data/training/scenarios/eval_scenarios.json save_path=datagen_debug checkpoint=data_info/xxx.json experiments.data_save_path_rel=data/Routes_Town01_Scenario3 experiments.DATAGEN=1 experiments.SAVE_SENSORS=1
# python leaderboard/scripts/run_evaluation.py experiments=detr eval.route_rel=leaderboard/data/training/routes/ll/debug.xml eval.scenarios_rel=leaderboard/data/training/scenarios/eval_scenarios.json save_path=simu checkpoint=data_info/xxx.json experiments.data_save_path_rel=data/Routes_Town01_Scenario3 experiments.DATAGEN=1 experiments.SAVE_SENSORS=1 +mmdet_cfg=/home/BuaaClass02/hrz/ndetr/projects/configs/detr3d/nu_carla.py

# CUDA_VISIBLE_DEVICES=0 python leaderboard/scripts/run_evaluation.py experiments=detr eval=longest6 save_path=output/debug_detr trafficManagerPort=8000 port=2010 +mmdet_cfg=/home/BuaaClass02/hrz/ndetr/projects/configs/detr3d/nu_carla.py +weight=/home/BuaaClass02/hrz/ndetr/work_dirs/detr_pl_route_command_50/epoch_13.pth resume=1
# CUDA_VISIBLE_DEVICES=0 python leaderboard/scripts/run_evaluation.py experiments=detr eval=longest6 save_path=output/debug_detr trafficManagerPort=8020 port=2020 resume=1 eval.route_rel=leaderboard/data/longest6/longest6_split/longest_weathers_0.xml +SAVE_SENSORS=1

# CUDA_VISIBLE_DEVICES=0 python leaderboard/scripts/run_evaluation.py experiments=detr eval=longest6 save_path=output/debug_detr2 trafficManagerPort=8020 port=2020 +mmdet_cfg=/home/BuaaClass02/hrz/ndetr/projects/configs/detr3d/nu_carla.py +weight=/home/BuaaClass02/hrz/ndetr/work_dirs/detr_pl_route_command_50/epoch_13.pth resume=1 eval.route_rel=leaderboard/data/longest6/longest6_split/longest_weathers_0.xml

# 这个用于debug
# CUDA_VISIBLE_DEVICES=0 python leaderboard/scripts/run_evaluation.py experiments=detr eval=longest6 save_path=output/debug_detr4 trafficManagerPort=8020 port=2020 +mmdet_cfg=/home/BuaaClass02/hrz/ndetr/projects/configs/detr3d/nu_carla.py +weight=/home/BuaaClass02/hrz/ndetr/work_dirs/detr_pl_route_command_50/epoch_15.pth eval.route_rel=leaderboard/data/longest6/longest6_split/longest_weathers_0.xml +SAVE_SENSORS=1

# 下面的
# CUDA_VISIBLE_DEVICES=2 python leaderboard/scripts/run_evaluation.py experiments=detr eval=longest6 save_path=output/simu_15 trafficManagerPort=8000 port=2000   resume=1

# CUDA_VISIBLE_DEVICES=2 python leaderboard/scripts/run_evaluation.py experiments=detr eval=longest6 save_path=output/simu_15_all +mmdet_cfg=projects/configs/detr3d/nu_carla_v2.py +weight=work_dirs/detr_pl_route_command_50/epoch_15.pth +SAVE_SENSORS=1 trafficManagerPort=8000 port=2000

CUDA_VISIBLE_DEVICES=2 python leaderboard/scripts/run_evaluation.py experiments=detr eval=longest6 save_path=output/simu_15_all +mmdet_cfg=projects/configs/detr3d/nu_carla_v2.py +weight=work_dirs/ndetr_hdmap/epoch_15.pth +SAVE_SENSORS=1 trafficManagerPort=8000 port=2000

# 

# 上面的
# CUDA_VISIBLE_DEVICES=0 python leaderboard/scripts/run_evaluation.py experiments=detr eval=longest6 save_path=output/metric_25 trafficManagerPort=8010 port=2010 +mmdet_cfg=/home/BuaaClass02/hrz/ndetr/projects/configs/detr3d/nu_carla.py +weight=/home/BuaaClass02/hrz/ndetr/work_dirs/detr_pl_route_command_50/epoch_25.pth +SAVE_SENSORS=1
