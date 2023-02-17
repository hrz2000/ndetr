import subprocess
import time
ports = ['00','10','20','30']
for idx, port in enumerate(ports):
    p = subprocess.Popen(f"CUDA_VISIBLE_DEVICES={idx} SDL_VIDEODRIVER=offscreen $CARLA_SERVER -carla-rpc-port=20{port} -nosound -opengl", shell=True)
    time.sleep(5)