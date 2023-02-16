# 设置代理
export http_proxy=http://10.1.8.5:32680/
export https_proxy=http://10.1.8.5:32680/
export HTTP_PROXY=http://10.1.8.5:32680/
export HTTPS_PROXY=http://10.1.8.5:32680/
# srun -p shlab-perceptionx apptainer run docker://hello-world # 可以运行
# srun -p shlab-perceptionx apptainer pull docker://carlasim/carla:0.9.10.1
# srun -p shlab-perceptionx apptainer run docker://carlasim/carla:0.9.10.1 #./CarlaUE4.sh --world-port=2000 -opengl
srun -p shlab-perceptionx apptainer run carla_0.9.10.1.sif #./CarlaUE4.sh --world-port=2000 -opengl

# docker18: docker run -it --rm -p 2000-2002:2000-2002 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 carlasim/carla:0.9.10.1 ./CarlaUE4.sh --world-port=2000 -opengl
# docker19: docker run -it --rm --net=host --gpus '"device=0"' carlasim/carla:0.9.10.1 ./CarlaUE4.sh --world-port=2000 -opengl

