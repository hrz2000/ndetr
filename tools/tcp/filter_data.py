import json
import os
import sys
import tqdm
from glob import glob
import shutil
import os.path as osp
import re

def remove_files(root, index, items = {"bev":".png", "meta":".json", "rgb":".png", "supervision":".npy"}, by_idxs=False):
	# items = {"measurements":".json", "rgb_front":".png"}
	items = {}
	sub_folders = list(os.listdir(root))
	for sub_folder in sub_folders:
		if len(list(os.listdir(os.path.join(root, sub_folder)))) == 0:
			break
		items[sub_folder] = "." + list(os.listdir(os.path.join(root, sub_folder)))[0].split(".")[-1]
	# {'topdown': '.png', 'attnmap': '.pkl', 'boxes': '.json', 'measurements': '.json', 'hdmap0': '.png', 'hdmap1': '.png', 'rgb': '.png'}
	if by_idxs==False:
		rm_sth = False
		for k, v in items.items():
			data_folder = os.path.join(root, k)
			total_len = len(os.listdir(data_folder)) # 从这个索引到最后
			for i in range(index, total_len): # 多次执行中间层可能有间断
				file_name = str(i).zfill(4) + v
				file_name = os.path.join(data_folder, file_name)
				try:
					os.remove(file_name)
					rm_sth = True
				except:
					pass
		return rm_sth
	else:
		for k, v in items.items():
			data_folder = os.path.join(root, k)
			for idx in index:
				file_name = str(idx).zfill(4) + v
				file_name = os.path.join(data_folder, file_name)
				try:
					os.remove(file_name)
				except:
					pass
    

def func(folder, checkpoint):
	data_folder=f'{folder}/Routes_{checkpoint}'
	result_file=f'{folder}/{checkpoint}.json'

	with open(os.path.join(result_file), 'r') as f:
		records = json.load(f)
	records = records["_checkpoint"]["records"]
	for index, record in enumerate(records):
		dirs = glob(f"{data_folder}/{checkpoint}_route{index}_*")
		dirs = sorted(dirs)
  
		# 可能有一个rourte跑多次的情况进行处理
		if len(dirs) != 1:
			# import pdb;pdb.set_trace()
			for i, d in enumerate(dirs):
				if i!=len(dirs)-1:
					# os.removedirs(d)
					shutil.rmtree(d)
     
		route_data_folder = dirs[-1]
		measlist=sorted(glob(f'{route_data_folder}/measurements/*.json'))
		length = len(measlist)
		if len(measlist) == 0:
			break
		last_meas=f'{route_data_folder}/measurements/{length-1:04d}.json'
		if record["scores"]["score_composed"] >= 100:
			continue

		if len(record["infractions"]["red_light"]) > 0 or \
			len(record["infractions"]["collisions_pedestrian"]) > 0 or \
			len(record["infractions"]["collisions_vehicle"]) > 0 or \
			len(record["infractions"]["collisions_layout"]) > 0 or \
			len(record["infractions"]["route_timeout"]) > 0 or \
			len(record["infractions"]["vehicle_blocked"]) > 0:
       
			idxs_dict = get_idxs_should_rm(record, route_data_folder, last_meas)
			idxs = []
			for k,v in idxs_dict.items():
				idxs.extend(v)
				meas = get_meas_list(route_data_folder, v)
				if len(meas) > 0:
					with open(f'filter_{k}.log', 'a') as f:
						f.write("\n".join(meas)+'\n')
			if len(idxs) > 0:
				meas = get_meas_list(route_data_folder, idxs)
				with open(f'filter_all.log', 'a') as f:
					f.write("\n".join(meas)+'\n')
def get_meas_list(route_data_folder, idxs):
    return [f"{route_data_folder}/vis/{idx:04d}.png" for idx in idxs]

def get_idxs_should_rm(record, route_data_folder, last_meas):
	red_light_events = record["infractions"]['red_light']
	vehicle_blockeds = record["infractions"]['vehicle_blocked']
	route_timeouts = record["infractions"]['route_timeout']
	colli_events = (record["infractions"]['collisions_pedestrian'])
	colli_events.extend(record["infractions"]['collisions_vehicle'])
	colli_events.extend(record["infractions"]['collisions_layout'])
 
	pattern = re.compile(r"at \(x=(.*?), y=(.*?), z=(.*?)\)") # r"\d"表示匹配任意一个数字
	colli_scenes = []
	for e_str in colli_events:
		match = re.search(pattern, e_str)
		x, y = float(match[1]), float(match[2])
		colli_scenes.append((x,y))	
  
	red_scenes = []
	for e_str in red_light_events:
		match = re.search(pattern, e_str)
		x, y = float(match[1]), float(match[2])
		red_scenes.append((x,y))

	block_scenes = []
	for e_str in vehicle_blockeds:
		# import pdb;pdb.set_trace()
		match = re.search(pattern, e_str)
		x, y = float(match[1]), float(match[2])
		block_scenes.append((x,y))

	timeout_scenes = []
	if len(route_timeouts) > 0:
		with open(os.path.join(last_meas), 'r') as f:
			meas = json.load(f)
			posx, posy = meas['pos_global']
		timeout_scenes.append((posx,posy))
  
	should_rm_idxs = {}
	should_rm_idxs['col'] = []
	should_rm_idxs['block'] = []
	should_rm_idxs['timeout'] = []
	should_rm_idxs['red'] = []
	# total_length=len()
	# 需要确保是连续的
	length = len(glob(f'{route_data_folder}/measurements/*.json'))
	for idx in range(length):
		# idx = int(os.path.splitext(os.path.basename(meas_path))[0])
		# print(idx)
		should_rm = False
		meas_file = f'{route_data_folder}/measurements/{idx:04d}.json'
		if not os.path.exists(meas_file):
			continue
		with open(os.path.join(meas_file), 'r') as f:
			meas = json.load(f)
			posx,posy = meas['pos_global']
			# posx,posy = int(posx), int(posy)

			for x,y in colli_scenes:
				if (posx >= x-5 and posx <= x+5 ) and (posy>=y-5 and posy <= y+5):
					should_rm = True
					# should_rm_idxs.append(idx)
					should_rm_idxs['col'].append(idx)
					break
			if should_rm == True:
				continue

			for x,y in red_scenes:
				if (posx >= x-10 and posx <= x+10 ) and (posy>=y-10 and posy <= y+10):
					should_rm = True
					should_rm_idxs['red'].append(idx)
					break
			if should_rm == True:
				continue

			for x,y in block_scenes:
				if (posx >= x-0.3 and posx <= x+0.3 ) and (posy>=y-0.3 and posy <= y+0.3):
					should_rm = True
					should_rm_idxs['block'].append(idx)
					# import pdb;pdb.set_trace()
					break
			if should_rm == True:
				continue

			for x,y in timeout_scenes:
				if (posx >= x-0.3 and posx <= x+0.3 ) and (posy>=y-0.3 and posy <= y+0.3):
					should_rm_idxs['timeout'].append(idx)
					break
	return should_rm_idxs

if __name__ == '__main__':
	folders = glob('output/plant_datagen/PlanT_data_1/*')
	for k in ['all','col','red','block','timeout']:
		with open(f'filter_{k}.log', 'w') as f:
			pass
	# folders = glob('output/datagen_l6/*')
	for folder in folders:
		# print(f"\nstart folder: {folder}************")
		checkpoints = glob(f'{folder}/*.json')
		for checkpoint in checkpoints:
			checkpoint = osp.splitext(osp.basename(checkpoint))[0]
			func(folder, checkpoint)