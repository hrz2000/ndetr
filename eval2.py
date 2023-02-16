# import mmcv
# mmcv.mkdir_or_exist('/home/BuaaClass07/hrz/ndetr/work_dirs/detr_gru_use_box/tf_logs')
import pynvml
import mmcv
import numpy as np
import os
from glob import glob
import subprocess, threading
import time
import argparse

parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('config', default='work_dirs/nu_carla_pgru_pretrain/nu_carla_pgru_pretrain.py', help='train config file path')
parser.add_argument('cuda', default=0, help='the dir to save logs and models')
parser.add_argument('port', default="00", help='the dir to save logs and models')
args = parser.parse_args()
cfg = args.config
cuda = args.cuda
port = args.port
path = 'leaderboard/data/longest6/longest6_split'
save_path = f"output/{os.path.basename(cfg)}_cuda_port"

p = subprocess.Popen(f"CUDA_VISIBLE_DEVICES={cuda} SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 $CARLA_SERVER --world-port=20{port} -opengl", shell=True)
time.sleep(10)

paths = os.listdir(path)
mask = []
for p in paths:
    cp = f'{save_path}/cp_{os.path.splitext(p)[0]}.json'
    need = 0
    if not os.path.exists(cp):
        need = 1
    try:
        f = mmcv.load(cp)
        rc = f['_checkpoint']['global_record']['scores']['score_route']
        if rc<10:
            need = 1
        else:
            need = 0
    except:
        need = 1
    mask.append(need)

npath = []
for i in range(len(mask)):
    if mask[i]==1:
        npath.append(paths[i])
paths = npath
print(len(paths), paths)

cmds = []
for idx, p in enumerate(paths):
    cmd = f'CUDA_VISIBLE_DEVICES={cuda} python leaderboard/scripts/run_evaluation.py experiments=detr eval=longest6 save_path={save_path} +mmdet_cfg={cfg} +weight=work_dirs/nu_carla_pgru_pretrain/epoch_15.pth +SAVE_SENSORS=1 eval.route_rel={path}/{p} trafficManagerPort=81{port} port=20{port} checkpoint={save_path}/cp_{os.path.splitext(p)[0]}.json +save_vis=1 +save_hdmap=1 +save_lane=0 resume=1\n'
    cmds.append(cmd)

for cmd in cmds:
    print(f"{cmd} start...")
    p = subprocess.Popen(f"{cmd}", shell=True)
    p.wait()
