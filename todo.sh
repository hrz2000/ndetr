CUDA_VISIBLE_DEVICES=2 SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 $CARLA_SERVER --world-port=2015 -opengl

CUDA_VISIBLE_DEVICES=3 python leaderboard/scripts/run_evaluation.py experiments=detr eval=longest6 save_path=output/no_route +mmdet_cfg=projects/configs/detr3d/no_route.py +weight=pretrain/cc_no_route.pth +SAVE_SENSORS=1 eval.route_rel=leaderboard/data/longest6/longest6_split/longest_weathers_32.xml trafficManagerPort=8305 port=2005 checkpoint=output/no_route/cp_longest_weathers_32.json +save_vis=1 +save_hdmap=1 +save_lane=0

# 2loss版本，save_path,mmcfg,checkpoint
CUDA_VISIBLE_DEVICES=2 python leaderboard/scripts/run_evaluation.py experiments=detr eval=longest6 save_path=output/no_route_2loss +mmdet_cfg=projects/configs/detr3d/no_route_2loss.py +weight=pretrain/cc_no_route_2loss.pth +SAVE_SENSORS=1 eval.route_rel=leaderboard/data/longest6/longest6_split/longest_weathers_32.xml trafficManagerPort=8310 port=2010 checkpoint=output/no_route_2loss/cp_longest_weathers_32.json +save_vis=1 +save_hdmap=1 +save_lane=0

python leaderboard/scripts/run_evaluation.py experiments=PlanTmedium3x eval=longest6 save_path=output/test_plant +SAVE_SENSORS=1 eval.route_rel=leaderboard/data/longest6/longest6_split/longest_weathers_32.xml trafficManagerPort=8020 port=2020 checkpoint_rel=cp_test32.json +save_vis=0 +save_hdmap=1 +save_lane=0 experiments.model_path=work_dirs/plant_ndetr_split viz=1

CUDA_VISIBLE_DEVICES=2 python leaderboard/scripts/run_evaluation.py experiments=PlanTmedium3x eval=longest6 save_path=output/myplant_one +SAVE_SENSORS=1 trafficManagerPort=8305 port=2005 +save_vis=1 +save_hdmap=1 +save_lane=0 experiments.model_path=work_dirs/plant_ndetr_split experiments.DATAGEN=1