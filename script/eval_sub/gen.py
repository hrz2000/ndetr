import mmcv
import time
import os.path as osp

num = 4
evalx = [[] for t in range(num)]

experiments='debug'
datagen=0
# eval_="longest6_debug"
eval_="longest6"
resume=1
timeout=60000
unblock=False
repetitions=1

config="projects/configs/detr3d/new/bs_box_wpattn_new.py"
checkpoint='pretrain/bugfix.pth'
# config="projects/configs/detr3d/new/bs_box_wpattn_new_weights.py"
# checkpoint='work_dirs/bs_box_wpattn_new_weights/2023-02-25_22:51:49/epoch_18.pth'
name = osp.splitext(osp.basename(config))[0]
save_path='output/output/detr_eval_split' + f"/{name}/" + time.strftime('%Y-%m-%d_%H:%M:%S')

track='MAP'

for idx in range(36):
    if idx == 4:
        break
    # todolist = [8,24,28,30,31,32,33,34,35]
    # if idx not in todolist:
    #     continue
    todolist = list(range(36))
    sub_idx = todolist.index(idx)
    cuda = sub_idx % 4
    res = sub_idx % num
    port = f"{res}0"

    route_rel=f'leaderboard/data/longest6/longest6_split/longest_weathers_{idx}.xml'
    scenarios_rel='leaderboard/data/longest6/eval_scenarios.json'
    
    base=f'CUDA_VISIBLE_DEVICES={cuda} python leaderboard/scripts/run_evaluation.py experiments={experiments} eval.route_rel={route_rel} eval.scenarios_rel={scenarios_rel}'
    # data_save_path_rel是save_path下面数据保存的地方，避免保存到同一个地方
    
    exp = f"experiments.DATAGEN={datagen} experiments.unblock={unblock}" # plant需要在exp里面设置model_path
    port_para = f"trafficManagerPort=82{port} port=20{port}"
    save_para = f"save_path={save_path}  experiments.data_save_path_rel=Route_{idx} checkpoint_rel={idx}.json +SAVE_SENSORS=1"
    mm_para = f"+mmdet_cfg={config} +weight={checkpoint}" # plant的时候也可以传入
    run_para = f"timeout={timeout} resume={resume} repetitions={repetitions} track={track}"
    cmd = [base, exp, port_para, save_para, mm_para, run_para]
    cmd = " ".join(cmd)
    
    bucket = sub_idx % num
    evalx[bucket].append(cmd)

# mmcv.save('\n'.join(eval1),file='script/eval_sub/eval1.sh',file_format='txt')
for i in range(num):
    with open(f'script/eval_sub/_eval{i}.sh', 'w') as f:
        f.write('\n'.join(evalx[i]))