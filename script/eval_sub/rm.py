import mmcv
import shutil
import os
path = '/home/BuaaClass02/hrz/ndetr/output/output/detr_eval_split/2023-02-17_11:50:27'
for i in range(36):
    json_file = f'{path}/{i}.json'
    info = mmcv.load(json_file)
    status = info['_checkpoint']['records'][0]['status']
    if 'Simulation crashed' in status:
        os.remove(json_file)
        # pass
    else:
        print(json_file)