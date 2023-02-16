import subprocess, threading
import mmcv
import os
from start import ports

# ls = mmcv.list_from_file('script/not_build.sh')
# ls = mmcv.list_from_file('script/datagen2.sh')
ls = mmcv.list_from_file('script/longest6.sh')
all_num = len(ls)
threads = len(ports)
for p in ports:
    with open(f'script/{p}.sh','w') as f:
        pass
# paths.sort()
# print(paths)
cmds = []
for idx, cmd in enumerate(ls):
    # if idx==threads:
    #     break
    port = ports[idx % threads]
    cuda = idx%4
    cmd = f'CUDA_VISIBLE_DEVICES={cuda} {cmd} trafficManagerPort=80{port} port=20{port} resume=1\n'
    with open(f'script/{port}.sh','a') as f:
        f.write(cmd)
    cmds.append(cmd)

for port in ports:
    def func(port):
        try:
            print(f"{port} start")
            name = f'script/{port}.sh'
            f2 = open(f"script/{port}.txt",'w')
            p = subprocess.Popen(f"bash {name}", stdout=f2, stderr=subprocess.STDOUT, shell=True)
            p.wait()
            f2.close()
            print(f"{port} completed")
        except KeyboardInterrupt:
            p.kill()
            f2.close()
    training_pin_thread = threading.Thread(target=func, args=[port])
    training_pin_thread.start()
