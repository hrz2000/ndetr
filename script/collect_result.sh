#!/bin/bash
# 注意，golbal_score中如果出现分数为-1，代表结果文件中没有该项指标
python ./script/collect_print.py --dir ./output/plant_datagen/PlanT_data_1 > ./script/result_collect.txt
# python ./script/collect_print.py --dir /home/BuaaClass02/hrz/plant/data/carla/PlanT_data_1 > ./script/result_collect.txt
