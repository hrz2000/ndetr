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

cmds=mmcv.list_from_file('script/datagen_sub/datagen.sh')

for idx in range(70): # TODO
    if idx<=20:
        continue
    res = idx % 4
    port = port_dict[res]
    cmd = cmds[idx] + f' trafficManagerPort=80{port} port=20{port}'
    evalx[res].append(cmd)

# mmcv.save('\n'.join(eval1),file='script/eval_sub/eval1.sh',file_format='txt')
for i in range(num):
    with open(f'script/datagen_sub/datagen{i}.sh', 'w') as f:
        f.write('\n'.join(evalx[i]))
