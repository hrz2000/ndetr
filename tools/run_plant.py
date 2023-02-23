import subprocess
import time
import os.path as osp

gpus=1
debug=False
# data_dir='output/plant_datagen/PlanT_data_1'
data_dir='output/plant_datagen2/PlanT_data_1'
data_dir='output/datagen_l6'
load_from='checkpoints/PlanT/3x/PlanT_medium/checkpoints/epoch\\=047.ckpt' # 没有的话会忽略,因为特殊字符的问题加载命令行里面会报错

p4 = subprocess.Popen(f"python training/PlanT/lit_train.py gpus={gpus} debug={debug} load_from='{load_from}' data_dir={data_dir}", shell=True)
p4.wait()