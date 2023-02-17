import subprocess
import time
# ports = ['00','10','20','30']
for idx, port in enumerate(range(4)):
    cuda = idx % 4
    port = f"{port}0"
    p = subprocess.Popen(f"CUDA_VISIBLE_DEVICES={cuda} SDL_VIDEODRIVER=offscreen $CARLA_SERVER -carla-rpc-port=20{port} -nosound -opengl", shell=True)
    time.sleep(3)