import json
import os
import sys
import tqdm
from glob import glob
import shutil
import os.path as osp

def remove_files(root, index, items = {"bev":".png", "meta":".json", "rgb":".png", "supervision":".npy"}):
	# items = {"measurements":".json", "rgb_front":".png"}
	items = {}
	sub_folders = list(os.listdir(root))
	for sub_folder in sub_folders:
		if len(list(os.listdir(os.path.join(root, sub_folder)))) == 0:
			break
		items[sub_folder] = "." + list(os.listdir(os.path.join(root, sub_folder)))[0].split(".")[-1]
	# {'topdown': '.png', 'attnmap': '.pkl', 'boxes': '.json', 'measurements': '.json', 'hdmap0': '.png', 'hdmap1': '.png', 'rgb': '.png'}
	for k, v in items.items():
		data_folder = os.path.join(root, k)
		total_len = len(os.listdir(data_folder)) # 从这个索引到最后
		for i in range(index, total_len):
			file_name = str(i).zfill(4) + v
			file_name = os.path.join(data_folder, file_name)
			os.remove(file_name)

def func(folder, checkpoint):
	data_folder=f'{folder}/Routes_{checkpoint}'
	result_file=f'{folder}/{checkpoint}.json'

	with open(os.path.join(result_file), 'r') as f:
		records = json.load(f)
	records = records["_checkpoint"]["records"]
	for index, record in enumerate(records):
		dirs = glob(f"{data_folder}/{checkpoint}_route{index}_*")
		dirs = sorted(dirs)
		if len(dirs) != 1:
			# import pdb;pdb.set_trace()
			for i, d in enumerate(dirs):
				if i!=len(dirs)-1:
					# os.removedirs(d)
					shutil.rmtree(d)
		route_data_folder = dirs[-1]
		total_length=len(glob(f'{route_data_folder}/measurements/*.json'))
		if record["scores"]["score_composed"] >= 100:
			continue
		# timeout or blocked, remove the last ones where the vehicle stops
		if len(record["infractions"]["route_timeout"]) > 0 or \
			len(record["infractions"]["vehicle_blocked"]) > 0:
				stop_index = 0
				for i in range(total_length-1, 0, -1):
					with open(os.path.join(route_data_folder, "measurements", str(i).zfill(4)) + ".json", 'r') as mf:
						speed = json.load(mf)["speed"]
						if speed > 0.1:
							stop_index = i
							break
				stop_index = min(total_length, stop_index + 20) # 在stop_index后面20帧就行
				print((route_data_folder, stop_index))
				remove_files(route_data_folder, stop_index)
		# collision or red-light
		elif len(record["infractions"]["red_light"]) > 0 or \
			len(record["infractions"]["collisions_pedestrian"]) > 0 or \
			len(record["infractions"]["collisions_vehicle"]) > 0 or \
			len(record["infractions"]["collisions_layout"]) > 0:
			stop_index = max(0, total_length-10) # 遇到这种碰撞的情况，删掉后10个
			print((route_data_folder, stop_index)) # 注意这个可能会导致重复的删除
			remove_files(route_data_folder, stop_index)

if __name__ == '__main__':
	folders = glob('/home/BuaaClass01/hrz/ndetr/output/plant_datagen/PlanT_data_1/*')
	for folder in folders:
		print(f"\nstart folder: {folder}************")
		checkpoints = glob(f'{folder}/*.json')
		for checkpoint in checkpoints:
			checkpoint = osp.splitext(osp.basename(checkpoint))[0]
			func(folder, checkpoint)