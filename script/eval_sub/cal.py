from glob import glob
import mmcv
import numpy as np

a = '/mnt/disk02/hrz/ndetr/output/output/detr_eval_split/bs_box_attnmap/2023-02-18_10:39:45'
# b = 'output/output/detr_eval_split/2023-02-17_10:55:27'
fa = glob(f"{a}/*json")
# fb = glob(f"{b}/*json")
print(len(fa))
# print(len(fb))
files = []
files.extend(fa)
# files.extend(fb)
# print(len(files))

composed=[]
penalty=[]
route=[]

for f in files:
    scores = mmcv.load(f)['_checkpoint']['global_record']['scores']
    rc = scores['score_composed']
    # if rc<10:
    #     continue
    composed.append(scores['score_composed'])
    penalty.append(scores['score_penalty'])
    route.append(scores['score_route'])

composed_arr = np.array(composed)
penalty_arr = np.array(penalty)
route_arr = np.array(route)
# import pdb;pdb.set_trace()
assert route_arr.shape[0] == 36

x,y,z=composed_arr.mean(),penalty_arr.mean(),route_arr.mean()
composed.append(x)
penalty.append(y)
route.append(z)
print(f"ds={composed},")
print(f"is_={penalty},")
print(f"rc={route}")
print(x,y,z)