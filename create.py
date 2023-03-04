import mmcv
mmcv.mkdir_or_exist('output') # sshfs
mmcv.mkdir_or_exist('test') # 用于保存本机的

# mmdetection3d checkpoints data work_dirs

# sshfs hrz@10.1.40.11:/mnt/disk02/hrz/ndetr/output ./output