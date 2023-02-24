import subprocess
import time
import os.path as osp

# experiments='detr'
# experiments='datagen'
# experiments='PlanTmedium3x'
# experiments='PlanTSubmission'
# experiments='PlanTSubmissionMap'
# experiments='PlanTSubmissionMap_datagen'
experiments='debug' # 需要设置datagen=0
# experiments='xxx'
is_ndetr = not ('PlanT' in experiments or experiments=='datagen')
if is_ndetr:
    config="projects/configs/detr3d/new/bs_box_wpattn_norouteandmap.py"
    checkpoint='work_dirs/bs_box_wpattn_norouteandmap/2023-02-24_14:30:12/epoch_18.pth'
    # config="projects/configs/detr3d/new/bs_box_wpattn_refine.py"
    # checkpoint='pretrain/cross_attn.pth'
else:
    config=experiments+'.py'
    checkpoint='no.pth'
    
if experiments=='PlanTSubmission':
    track='SENSORS'
else:
    track='MAP'
    
datagen=0
# eval_="longest6_debug"
eval_="longest6"
port="25"
resume=1
timeout=200
unblock=False
repetitions=1


save_path=osp.join('./output/output', osp.splitext(osp.basename(config))[0])+ "/" + time.strftime('%Y-%m-%d_%H:%M:%S')

p = subprocess.Popen(f"SDL_VIDEODRIVER=offscreen $CARLA_SERVER -carla-rpc-port=20{port} -nosound -opengl", shell=True)

time.sleep(10)

t1 = time.time()

base = f"python leaderboard/scripts/run_evaluation.py experiments={experiments} eval={eval_}"
exp = f"experiments.DATAGEN={datagen} experiments.unblock={unblock}" # plant需要在exp里面设置model_path
port_para = f"trafficManagerPort=80{port} port=20{port}"
save_para = f"save_path={save_path} +SAVE_SENSORS=1"
mm_para = f"+mmdet_cfg={config} +weight={checkpoint}" # plant的时候也可以传入
run_para = f"timeout={timeout} resume={resume} repetitions={repetitions} track={track}"
cmd = [base, exp, port_para, save_para, mm_para, run_para]
cmd = " ".join(cmd)

# cmd = 'python tools/runcarla.py' ##xx

p3 = subprocess.Popen(cmd, shell=True)
p3.wait()

# if is_ndetr:
#     p4 = subprocess.Popen(f"python tov.py --re '{save_path}/data_save_path/*/vis/*' -s {save_path}/data_save_path/a.mp4", shell=True)
#     p4.wait()
#     print(f"video: {save_path}/data_save_path/a.mp4")
# else:
#     p4 = subprocess.Popen(f"python tov.py --re './vis/*' -s ./a.mp4", shell=True)
#     p4.wait()
#     print(f"video: ./a.mp4")

subprocess.Popen(f"fuser -k /dev/nvidia*", shell=True)
print(f"over, use time {time.time()-t1}s")
