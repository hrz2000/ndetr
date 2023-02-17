import os
from glob import glob
import subprocess, threading

path = 'leaderboard/data/longest6/longest6_split'
save_path = "output/myplant_l6"
from start import ports
threads = len(ports)
for p in ports:
    with open(f'script/{p}.sh','w') as f:
        pass
# paths.sort()
# print(paths)
# cmds = []
for idx, p in enumerate(os.listdir(path)):
    # if idx==threads:
    #     break
    port = ports[idx % threads]
    cuda = idx%2
    cmd = f'CUDA_VISIBLE_DEVICES={cuda} python leaderboard/scripts/run_evaluation.py experiments=datagen eval=longest6 save_path={save_path}  +SAVE_SENSORS=1 eval.route_rel={path}/{p} trafficManagerPort=80{port} port=20{port} checkpoint_rel=cp_{os.path.splitext(p)[0]}.json +save_vis=0 +save_hdmap=1 +save_lane=0\n'
    with open(f'script/{port}.sh','a') as f:
        f.write(cmd)
    # cmds.append(cmd)

for port in ports:
    def func(port):
        try:
            print(f"{port} start")
            name = f'script/{port}.sh'
            f2 = open(f"script/{port}.txt",'w')
            p = subprocess.Popen(f"bash {name}", stdout=f2, stderr=subprocess.STDOUT, shell=True)
            p.wait()
            f2.close()
            print(f"{port} completed")
        except KeyboardInterrupt:
            p.kill()
            f2.close()
    training_pin_thread = threading.Thread(target=func, args=[port])
    training_pin_thread.start()
