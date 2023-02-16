import subprocess

for i in range(4):
    with open(f'script/datagen_sub/datagen{i}.txt','w') as f:
        p = subprocess.Popen(f"CUDA_VISIBLE_DEVICES={i} bash script/datagen_sub/datagen{i}.sh",stdout=f,shell=True)