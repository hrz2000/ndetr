# conda deactivate
pwd_root=$(pwd)

cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
rm -f ./etc/conda/activate.d/env_vars.sh
rm -f ./etc/conda/deactivate.d/env_vars.sh
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh

carla_root=/mnt/disk02/hrz/ndetr/carla

echo "export CARLA_ROOT=${carla_root}" >> ./etc/conda/activate.d/env_vars.sh
echo "export CARLA_SERVER=\${CARLA_ROOT}/CarlaUE4.sh" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=\$PYTHONPATH:\${CARLA_ROOT}/PythonAPI" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=\$PYTHONPATH:\${CARLA_ROOT}/PythonAPI/carla" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=\$PYTHONPATH:\$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=\$PYTHONPATH:leaderboard" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=\$PYTHONPATH:scenario_runner" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=mmdetection3d:\$PYTHONPATH" >> ./etc/conda/activate.d/env_vars.sh

echo "unset CARLA_ROOT" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset CARLA_SERVER" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset PYTHONPATH" >> ./etc/conda/deactivate.d/env_vars.sh

# cd $pwd_root
# rm -f ./carla_agent_files/config/user/$USER.yaml
# rm -f ./training/config/user/$USER.yaml
# touch ./carla_agent_files/config/user/$USER.yaml
# touch ./training/config/user/$USER.yaml
# echo "working_dir: $pwd_root" >> ./carla_agent_files/config/user/$USER.yaml
# echo "working_dir: $pwd_root" >> ./training/config/user/$USER.yaml
# echo "carla_path: $pwd_root/carla" >> ./carla_agent_files/config/user/$USER.yaml

# conda deactivate
# conda activate plant