import os
from glob import glob
import json
import wandb

def update_wandb_at_curdir():
    log_jsons = sorted(glob('./*.log.json'))
    assert len(log_jsons) > 0
    exp_name = sorted(glob('*.py'))[0]

    wandb.init(
        project="your-awesome-project",
        name=exp_name)
    for log_json in log_jsons:
        with open(log_json, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'mode' in data:
                    mode = data.pop('mode')
                    if mode == 'train':
                        epoch = data.pop('epoch')
                        iter = data.pop('iter')
                        # NOTICE: 2810 is iter per epoch
                        step = (epoch-1) * 2810 + iter
                        lr = data.pop('lr')
                        data.pop('memory')
                        data.pop('data_time')
                        # key: train/loss_cls will add a panel named 'train'
                        data = {mode + '/' + k: v for k, v in data.items()}
                        data['learning_rate'] = lr
                        wandb.log(data, step=step)
                    elif mode == 'val':
                        data.pop('epoch')
                        data.pop('iter')
                        data.pop('lr')
                        data = {mode + '/' + k: v for k, v in data.items()}
                        # NOTICE: 67441 is epoch * iter per epoch + 1
                        wandb.log(data, step=67441)
    # wandb.finish()

if __name__ == '__main__':
    dirs = [
        'work_dirs/bs_box_attnmap/2023-02-16_12:36:22',
    ]
    for dir in dirs:
        os.chdir(dir)
        update_wandb_at_curdir()
        os.chdir('..')
