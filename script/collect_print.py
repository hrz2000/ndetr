import json 
import argparse
import glob 
import os

dataset2cp = {'coke_s3_dataset': ['Town01_Scenario3.json', 'Town02_Scenario3.json', 'Town03_Scenario3.json', 'Town04_Scenario3.json', 'Town05_Scenario3.json', 'Town06_Scenario3.json', 'Town07_Scenario3.json', 'Town10HD_Scenario3.json'], 'cycl_s4_dataset': ['Town01_Scenario4.json', 'Town02_Scenario4.json', 'Town03_Scenario4.json', 'Town04_Scenario4.json', 'Town05_Scenario4.json', 'Town06_Scenario4.json', 'Town07_Scenario4.json', 'Town10HD_Scenario4.json'], 'dirt_s1_dataset': ['Town01_Scenario1.json', 'Town02_Scenario1.json', 'Town03_Scenario1.json', 'Town04_Scenario1.json', 'Town05_Scenario1.json', 'Town06_Scenario1.json', 'Town07_Scenario1.json', 'Town10HD_Scenario1.json'], 'int_l_s8_dataset': ['Town01_Scenario8.json', 'Town02_Scenario8.json', 'Town03_Scenario8.json', 'Town04_Scenario8.json', 'Town05_Scenario8.json', 'Town06_Scenario8.json', 'Town07_Scenario8.json', 'Town10HD_Scenario8.json'], 'int_r_s9_dataset': ['Town01_Scenario9.json', 'Town02_Scenario9.json', 'Town03_Scenario9.json', 'Town04_Scenario9.json', 'Town05_Scenario9.json', 'Town06_Scenario9.json', 'Town07_Scenario9.json', 'Town10HD_Scenario9.json'], 'int_s_s7_dataset': ['Town01_Scenario7.json', 'Town02_Scenario7.json', 'Town03_Scenario7.json', 'Town04_Scenario7.json', 'Town05_Scenario7.json', 'Town06_Scenario7.json', 'Town07_Scenario7.json', 'Town10HD_Scenario7.json'], 'int_u_s10_dataset': ['Town03_Scenario10.json', 'Town04_Scenario10.json', 'Town05_Scenario10.json', 'Town06_Scenario10.json', 'Town07_Scenario10.json', 'Town10HD_Scenario10.json'], 'll_dataset': ['Town04_ll.json', 'Town05_ll.json', 'Town06_ll.json'], 'lr_dataset': ['Town03_lr.json', 'Town04_lr.json', 'Town05_lr.json', 'Town06_lr.json', 'Town10HD_lr.json'], 'rl_dataset': ['Town03_rl.json', 'Town04_rl.json', 'Town05_rl.json', 'Town06_rl.json', 'Town10HD_rl.json'], 'rr_dataset': ['Town04_rr.json', 'Town05_rr.json', 'Town06_rr.json']}
nd = {}

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="/home/BuaaClass07/hrz/ndetr/output/plant_datagen/PlanT_data_1")

args = parser.parse_args()
root_path = args.dir

print(f"target dir: {root_path}")
# os.chdir(root_path)

dataset_list = glob.glob(f"{str(args.dir)}/*dataset")
dataset_list.sort()
print(f"all_data_set:{dataset_list}")
not_build = []

def get_score(global_score):
    try:
        global_score = global_score[0]
    except:
        pass
    score_composed = global_score.get('score_composed', -1)
    score_penalty = global_score.get('score_penalty', -1)
    score_route = global_score.get('score_route', -1)
    return score_composed, score_penalty, score_route

for now_path in dataset_list:
    dataset = os.path.basename(now_path)
    print(f"\n{dataset}")
    res_list = glob.glob(f"{now_path}/*.json")
    res_list.sort()
    
    cps = {t:0 for t in dataset2cp[dataset]}

    for res in res_list:
        change_content = {}
        with open(res, 'r') as f:
            cp_name = os.path.basename(res)
            cps[cp_name] = 1
            content = json.load(f)

            global_score = content['_checkpoint']['global_record'].get('scores', dict()),
            score_composed ,score_penalty ,score_route = get_score(global_score)

            progress = content['_checkpoint']['progress']

            if score_composed>=90 and progress[0] == progress[1]:
                # print(f"============{cp_name}_ok============")
                continue
            else:
                print(f"============{cp_name}_unok============")
                # print('\t\t\t\tcomposed penalty route')

            if len(progress)==2 and progress[0] == progress[1]:
                print(f'status: completed')
            else:
                print(f'status: uncompleted, {progress}')
                cps[cp_name] = 0
                
            print(f'global_scores: \t{score_composed:.2f}\t{score_penalty:.2f}\t{score_route:.2f}')
            records_scores = []
            for record in content['_checkpoint']['records']:
                idx = record['index']
                score_composed ,score_penalty ,score_route = get_score(record['scores'])
                if score_composed>=90:
                    continue
                print(f'records_{idx}:  \t{score_composed:.2f}\t{score_penalty:.2f}\t{score_route:.2f}')
    
    for k in cps:
        if cps[k] == 0:
            print(k, "not built")
            not_build.append(k)

print("\n###############\nnot_build:")
print(not_build)
import mmcv
ls = mmcv.list_from_file('script/datagen2.sh')
cmds = []
for t in not_build:
    for l in ls:
        if t in l:
            cmds.append(l)
            continue
with open('script/not_build.sh', 'w') as f:
    f.write('\n'.join(cmds))

print()
print('\n'.join(cmds))


print("\n#######################################################\n")

# dataset_list = glob.glob(f"{str(args.dir)}/*dataset/Route*/*route*")
# dataset_list.sort()
# import re
# # for d in dataset_list:

# print("\n".join(dataset_list))

