from glob import glob
import mmcv
files = glob('output/pretrain/*.json')
lf = len(files)
print(lf)
composed=0
penalty=0
route=0
length=0

for f in files:
    try:
        scores = mmcv.load(f)['_checkpoint']['global_record']['scores']
        rc = scores['score_composed']
        if rc<10:
            continue
        composed += scores['score_composed']
        penalty += scores['score_penalty']
        route += scores['score_route']
        length+=1
    except:
        pass

print(length)
for i in [composed,penalty,route]:
    print(i/length)
# 83.40620539922583
# 0.9027777777777778
# 92.00982801922358 专家

# 14
# 43.24282531299293
# 0.5908571428571429
# 75.03654304081572 忽略掉RC<10的，

# 18
# 46.123993980810184
# 0.60525
# 77.11937951995691