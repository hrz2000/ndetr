import json 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="/home/BuaaClass07/hrz/ndetr/output/plant_datagen/PlanT_data_1/coke_s3_dataset")

args = parser.parse_args()
root_path = args.dir

print(f"target dir: {root_path}")
res_path = 'collect'


import os
os.chdir(root_path)

import glob 
res_list = glob.glob("*.json")
res_list.sort()

res_content = []
for res in res_list:
    change_content = {}
    with open(res, 'r') as f:
        content = json.load(f)

        records_scores = []
        for record in content['_checkpoint']['records']:
            records_scores.append({
                "index": record['index'],
                "scores": record['scores']
            })
        change_content = {
            "file_name": res,
            "global_scores": content['_checkpoint']['global_record'].get('scores', {}),
            "records_scores": records_scores
        }
        res_content.append(change_content)    

if not os.path.exists('collect'):
    os.mkdir('collect')

real_res_path = os.path.join('collect', root_path.split('/')[-1]+'_result.json') 
with open(real_res_path, 'w') as f:
    json.dump({
        'results': res_content
    } ,f,  indent=4)
print(f"result is collected at: {os.path.join(root_path, real_res_path)}")

        