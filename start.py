import subprocess, threading, time
import mmcv
mmcv.mkdir_or_exist('carla_log')
def func(port, idx):
    print(f"carla_{port} start")
    cuda = idx%4
    f2 = open(f"carla_log/carla_{port}.txt",'w')
    p = subprocess.Popen(f"CUDA_VISIBLE_DEVICES={cuda} SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 $CARLA_SERVER --world-port=20{port} -opengl", stdout=f2, stderr=subprocess.STDOUT, shell=True)
    p.wait()
    f2.close()
    print(f"carla_{port} completed")

ports = ["25","75","15","05","00","12","20","30","40","50","60","70"]
# ports = ["30","40"]#,"50", "60","70","80"]
if __name__ == '__main__':
    for idx, port in enumerate(ports):
        training_pin_thread = threading.Thread(
            target=func, args=[port, idx])
        training_pin_thread.start()
        time.sleep(5)
