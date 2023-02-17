import subprocess

# for i in range(4):
#     with open(f'script/datagen_sub/_datagen{i}.txt','w') as f:
#         p = subprocess.Popen(f"CUDA_VISIBLE_DEVICES={i} bash script/datagen_sub/_datagen{i}.sh",stdout=f,shell=True)

for i in range(4):
    # with open(f'script/datagen_sub/__datagen{i}.txt','w') as f:
    p = subprocess.Popen(f"bash script/eval_sub/_eval{i}.sh > script/datagen_sub/__eval{i}.txt",shell=True)