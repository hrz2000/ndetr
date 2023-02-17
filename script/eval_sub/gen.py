import mmcv
import time

num = 4
evalx = [[] for t in range(num)]

experiments='debug'
datagen=0
# eval_="longest6_debug"
eval_="longest6"
resume=1
timeout=200
unblock=False
repetitions=1
save_path='output/output/detr_eval_split' + "/" + time.strftime('%Y-%m-%d_%H:%M:%S')
config="projects/configs/detr3d/new/bs_box.py"
checkpoint='work_dirs/bs_box/2023-02-15_21:51:35/epoch_18.pth'
track='MAP'
port_dict = {0:'00',1:'10',2:'20',3:'30'}

for idx in range(36):
    res = idx % 4
    port = port_dict[res]
    route_rel=f'leaderboard/data/longest6/longest6_split/longest_weathers_{idx}.xml'
    scenarios_rel='leaderboard/data/longest6/eval_scenarios.json'
    
    base=f'CUDA_VISIBLE_DEVICES={res} python leaderboard/scripts/run_evaluation.py experiments={experiments} eval.route_rel={route_rel} eval.scenarios_rel={scenarios_rel}'
    # data_save_path_rel是save_path下面数据保存的地方，避免保存到同一个地方
    
    exp = f"experiments.DATAGEN={datagen} experiments.unblock={unblock}" # plant需要在exp里面设置model_path
    port_para = f"trafficManagerPort=80{port} port=20{port}"
    save_para = f"save_path={save_path}  experiments.data_save_path_rel=Route_{idx} checkpoint_rel={idx}.json +SAVE_SENSORS=1"
    mm_para = f"+mmdet_cfg={config} +weight={checkpoint}" # plant的时候也可以传入
    run_para = f"timeout={timeout} resume={resume} repetitions={repetitions} track={track}"
    cmd = [base, exp, port_para, save_para, mm_para, run_para]
    cmd = " ".join(cmd)
    
    evalx[res].append(cmd)

# mmcv.save('\n'.join(eval1),file='script/eval_sub/eval1.sh',file_format='txt')
for i in range(num):
<<<<<<< HEAD
    with open(f'script/eval_sub/eval{i}.sh', 'w') as f:
=======
    with open(f'script/eval_sub/_eval{i}.sh', 'w') as f:
>>>>>>> b46f7ebbe7c44d2a73c2fcbc87bae9c47a21b9d7
        f.write('\n'.join(evalx[i]))