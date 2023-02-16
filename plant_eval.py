import os
from glob import glob
import subprocess, threading
import mmcv

path = 'leaderboard/data/longest6/longest6_split'
save_path = "output/myplant_l6"
from start import ports
threads = len(ports)
for port in ports:
    with open(f'carla_log/py_{port}.txt','w') as f:
        pass
    with open(f'carla_log/sh_{port}.sh','w') as f:
        pass
# paths.sort()
# print(paths)
# cmds = []
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

for idx, p in enumerate(paths):
    # if idx==threads:
    #     break
    port = ports[idx % threads]
    cuda = idx%2
    cmd = f'CUDA_VISIBLE_DEVICES={cuda} python leaderboard/scripts/run_evaluation.py experiments=PlanTmedium3x eval=longest6 save_path={save_path} +SAVE_SENSORS=1 eval.route_rel={path}/{p} trafficManagerPort=80{port} port=20{port} checkpoint_rel=cp_{os.path.splitext(p)[0]}.json +save_vis=1 +save_hdmap=1 +save_lane=0 experiments.model_path=work_dirs/plant_ndetr_split experiments.DATAGEN=1\n'# resume=1
    with open(f'carla_log/sh_{port}.sh','a') as f:
        f.write(cmd)
    # cmds.append(cmd)

for port in ports:
    def func(port):
        try:
            print(f"{port} start")
            name = f'carla_log/sh_{port}.sh'
            # txt = f"carla_log/py_{port}.txt"
            f2 = open(f"carla_log/py_{port}.txt",'w')
            p = subprocess.Popen(f"bash {name}", stdout=f2, stderr=subprocess.STDOUT, shell=True)#
            p.wait()
            f2.close()
            print(f"{port} completed")
        except KeyboardInterrupt:
            p.kill()
            f2.close()
    training_pin_thread = threading.Thread(target=func, args=[port])
    training_pin_thread.start()
