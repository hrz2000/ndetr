import subprocess
cmds = ['python tools/run_carla_ndetr.py --config projects/configs/detr3d/new/baseline_weight_box_velo.py --checkpoint pretrain/baseline_weight_box_velo_4gpu.pth --port 84 --eval longest6_sub1',
'python tools/run_carla_ndetr.py --config projects/configs/detr3d/new/baseline_weight_box_velo.py --checkpoint pretrain/baseline_weight_box_velo_4gpu.pth --port 74 --eval longest6_sub2',
'python tools/run_carla_ndetr.py --config projects/configs/detr3d/new/baseline_weight_box_velo.py --checkpoint pretrain/baseline_weight_box_velo_4gpu.pth --port 64 --eval longest6_sub3',
'python tools/run_carla_ndetr.py --config projects/configs/detr3d/new/baseline_weight_box_velo.py --checkpoint pretrain/baseline_weight_box_velo_4gpu.pth --port 54 --eval longest6_sub4']
for cmd in cmds:
    subprocess.Popen(cmd, shell=True)
print("start over")
