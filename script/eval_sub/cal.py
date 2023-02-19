from glob import glob
import mmcv

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

composed=0
penalty=0
route=0
length=0

for f in files:
    try:
        scores = mmcv.load(f)['_checkpoint']['global_record']['scores']
        rc = scores['score_composed']
        # if rc<10:
        #     continue
        composed += scores['score_composed']
        penalty += scores['score_penalty']
        route += scores['score_route']
        length+=1
    except:
        pass

print(length)
for i in [composed,penalty,route]:
    print(i/length)