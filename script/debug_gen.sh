# python leaderboard/scripts/run_evaluation.py experiments=datagen eval.route_rel=leaderboard/data/training/routes/ll/debug.xml eval.scenarios_rel=leaderboard/data/training/scenarios/eval_scenarios.json save_path=datagen_debug checkpoint=data_info/xxx.json experiments.data_save_path_rel=data/Routes_Town01_Scenario3 experiments.DATAGEN=1 experiments.SAVE_SENSORS=1
# python leaderboard/scripts/run_evaluation.py experiments=detr eval.route_rel=leaderboard/data/training/routes/ll/debug.xml eval.scenarios_rel=leaderboard/data/training/scenarios/eval_scenarios.json save_path=simu checkpoint=data_info/xxx.json experiments.data_save_path_rel=data/Routes_Town01_Scenario3 experiments.DATAGEN=1 experiments.SAVE_SENSORS=1 +mmdet_cfg=/home/BuaaClass02/hrz/ndetr/projects/configs/detr3d/nu_carla.py

CUDA_VISIBLE_DEVICES=0 python leaderboard/scripts/run_evaluation.py experiments=datagen eval=longest6 save_path=output/debug_gen2 trafficManagerPort=8000 port=2010 resume=1 +SAVE_SENSORS=1 eval.route_rel=leaderboard/data/longest6/longest6_split/longest_weathers_0.xml
# CUDA_VISIBLE_DEVICES=0 python leaderboard/scripts/run_evaluation.py experiments=detr eval=longest6 save_path=output/debug_detr trafficManagerPort=8020 port=2020 resume=1 eval.route_rel=leaderboard/data/longest6/longest6_split/longest_weathers_0.xml +SAVE_SENSORS=1

# CUDA_VISIBLE_DEVICES=0 python leaderboard/scripts/run_evaluation.py experiments=datagen eval=longest6 save_path=output/debug_datagen trafficManagerPort=8000 port=2010 +mmdet_cfg=/home/BuaaClass02/hrz/ndetr/projects/configs/detr3d/nu_carla.py +weight=/home/BuaaClass02/hrz/ndetr/work_dirs/detr_pl_route_command_50/epoch_13.pth resume=1 eval.route_rel=leaderboard/data/longest6/longest6_split/longest_weathers_0.xml
