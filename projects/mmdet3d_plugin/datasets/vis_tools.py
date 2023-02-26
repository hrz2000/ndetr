import numpy as np
from os import path as osp
import numpy as np
import copy

from mmdet3d.core.visualizer.image_vis import draw_camera_bbox3d_on_img
import math
from PIL import Image, ImageDraw
import numpy as np
import mmcv
import cv2

num2str = {-1:"none",1:"left",2:'right',3:'straight',4:'lanefollow',5:'changelaneleft',6:'changelaneright'}

def project_pts_on_img(points,
                       raw_img,
                       lidar2img_rt,
                       color=None,
                       thickness=-1):
    img = raw_img.copy()
    num_points = points.shape[0]
    pts_4d = np.concatenate([points[:, :3], np.ones((num_points, 1))], axis=-1)
    # import pdb;pdb.set_trace()
    pts_2d = pts_4d @ lidar2img_rt.T

    # cam_points is Tensor of Nx4 whose last column is 1
    # transform camera coordinate to image coordinate
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    fov_inds = ((pts_2d[:, 0] < img.shape[1])
                & (pts_2d[:, 0] >= 0)
                & (pts_2d[:, 1] < img.shape[0])
                & (pts_2d[:, 1] >= 0))

    imgfov_pts_2d = pts_2d[fov_inds, :3]  # u, v, d

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pts_2d[i, 2]
        cv2.circle(
            img,
            center=(int(np.round(imgfov_pts_2d[i, 0])),
                    int(np.round(imgfov_pts_2d[i, 1]))),
            radius=5,
            color=color,
            thickness=thickness,
        )
    return img

def wp23d(reference_points, h=0):
    bs, num, nd = reference_points.shape
    if nd == 2:
        n_reference_points = reference_points.new_zeros((bs, num, 3))
        n_reference_points[:, :, -1] = h # 需要在lidar系下
        n_reference_points[...,:2] = reference_points
        reference_points = n_reference_points
    return reference_points

def create_front(pred_pts_bbox, gt_pts_bbox, front_img, cam2img):
    # 预测里面没有ego，gt里面必须要有ego
    if pred_pts_bbox!=None and 'boxes_3d' in pred_pts_bbox:
        front_img = draw_camera_bbox3d_on_img(
            pred_pts_bbox['boxes_3d'], # 这个是instance3d
            raw_img=front_img,
            cam2img=cam2img,
            img_metas=None,
            color=(255, 0, 0),
            thickness=1)
    if pred_pts_bbox!=None and 'attrs_3d' in pred_pts_bbox:
        wp = pred_pts_bbox['attrs_3d'] # 是lidar系
        wp = wp23d(wp[None], -2.5)[0]
        wp[:, 1] *= -1
        front_img = project_pts_on_img(
            wp, # 这个是instance3d
            raw_img=front_img,
            lidar2img_rt=cam2img,
            color=(255, 0, 0),
            thickness=1)
    if gt_pts_bbox!=None and 'boxes_3d' in gt_pts_bbox:
        front_img = draw_camera_bbox3d_on_img(
            gt_pts_bbox['boxes_3d'][1:], # 这个也是instance3d
            raw_img=front_img,
            cam2img=cam2img,
            img_metas=None,
            color=(0, 255, 0),
            thickness=1)
    return front_img

# def vis_bev_view(pred_pts_bbox, gt_pts_bbox, PIXELS_PER_METER = 5, size = (300,300)):
#     img = Image.new('RGB', size)
#     if gt_pts_bbox['gt_bev'] is not None:
#         img = Image.fromarray(mmcv.imread(gt_pts_bbox['gt_bev']))
#         size = img.size
#     origin = (size[0]//2, size[1]//2)
#     draw = ImageDraw.Draw(img)

#     # point = origin
#     # draw.ellipse(((point[0]-1,point[1]-1),(point[0]+1,point[1]+1)), fill='red', width=4) # center

#     create_bev(draw, pred_pts_bbox, gt_pts_bbox, PIXELS_PER_METER, origin)

#     img = np.array(img)[:,:,::-1]
#     return img

# def bev_draw_line(draw, wp, PIXELS_PER_METER, origin, is_grey=False, front=0, width=0):
#     wp[:,0] += 1.3 # 从lidar到ego系
#     wp[:,0] += front
#     args = []
#     for idp, p in enumerate([wp[0],wp[3]]):
#         x = p[1]*PIXELS_PER_METER + origin[1]
#         y = -p[0]*PIXELS_PER_METER + origin[0]
#         args.append(x)
#         args.append(y)
#     draw.line(args,fill=1,width=width*PIXELS_PER_METER)

# def bev_draw_box(draw, pred_pts_bbox, gt_pts_bbox, PIXELS_PER_METER, origin, is_grey=False):
#     for box_type, (result, c) in enumerate([(gt_pts_bbox,"chartreuse"), (pred_pts_bbox,"dodgerblue")]):
#         if result is None:
#             continue
#         arrs = copy.deepcopy(result['boxes_3d'].tensor.detach().cpu().numpy())
#         labels = result['labels_3d']
#         arrs[:,0] += 1.3

#         for idx, arr in enumerate(arrs):#(9,)
#             # label = labels[idx]
#             # name = classes[label]
#             box = {}
#             box['position']=(arr[0],arr[1],arr[2])
#             box['extent']=(arr[3+2],arr[3+0],arr[3+1])
#             box['yaw']=arr[6]
#             box['speed']=np.sqrt(arr[7]**2+arr[8]**2)

#             x = -box['position'][1]*PIXELS_PER_METER + origin[1]#y是向右
#             y = -box['position'][0]*PIXELS_PER_METER + origin[0]#x是向前 本来就应该是-
#             yaw = box['yaw']
#             extent_x = box['extent'][2]*PIXELS_PER_METER/2
#             extent_y = box['extent'][1]*PIXELS_PER_METER/2
            
#             p1, p2, p3, p4 = get_coords_BB(x, y, yaw, extent_x, extent_y)

#             if is_grey:
#                 draw.polygon((p1, p2, p3, p4), fill=1)
#             else:
#                 # point = (int(x),int(y))
#                 # draw.text(point, f"{np.array(arr)[:2].astype(int)}", fill='white', width=2) # center
#                 if box_type==0:
#                     draw.polygon((p1, p2, p3, p4), fill=c, outline='black')
#                 else:
#                     draw.polygon((p1, p2, p3, p4), outline=c)

#                 if 'speed' in box:
#                     vel = box['speed']*3 #/3.6 # in m/s # just for visu
#                     endx1, endy1, endx2, endy2 = get_coords(x, y, yaw, vel)
#                     fill_c = 'orange' if box_type==0 else 'royalblue'
#                     draw.line((endx1, endy1, endx2, endy2), fill=fill_c, width=1)

def create_fut_bev(pred_pts_bbox, gt_pts_bbox, only_box_for_col_det=None, PIXELS_PER_METER = 5, size = (300,300)):
    if gt_pts_bbox['topdown'] is not None:
        img = Image.fromarray(mmcv.imread(gt_pts_bbox['topdown'])) # TODO: GT_BEV
        size = img.size
    else:
        img = Image.new('RGB', size)
    draw = ImageDraw.Draw(img)
    origin = (size[0]//2, size[1]//2)
    batch_fut_box = pred_pts_bbox.get('fut_boxes', None) # (bs, num_box, ndim=10)
    if batch_fut_box is None:
        return None
    for batch_idx, fut_box in enumerate(batch_fut_box):
        arrs = fut_box.detach().cpu().numpy()
        # arrs[:,0] += 1.3 # fut_box本身就是ego系

        for idx, arr in enumerate(arrs): # (9,)
            box = {}
            box['position']=(arr[0],arr[1],arr[2])
            box['extent']=(arr[3+2],arr[3+0],arr[3+1])
            box['yaw']=arr[6]
            box['speed']=np.sqrt(arr[7]**2+arr[8]**2)

            x = -box['position'][1]*PIXELS_PER_METER + origin[1]#y是向右
            y = -box['position'][0]*PIXELS_PER_METER + origin[0]#x是向前 本来就应该是-
            yaw = box['yaw']
            extent_x = box['extent'][2]*PIXELS_PER_METER/2
            extent_y = box['extent'][1]*PIXELS_PER_METER/2
            
            p1, p2, p3, p4 = get_coords_BB(x, y, yaw, extent_x, extent_y)

            # point = (int(x),int(y))
            # draw.text(point, f"{np.array(arr)[:2].astype(int)}", fill='white', width=2) # center
            draw.polygon((p1, p2, p3, p4), outline='blue')

            if 'speed' in box:
                vel = box['speed']*3 #/3.6 # in m/s # just for visu
                endx1, endy1, endx2, endy2 = get_coords(x, y, yaw, vel)
                fill_c = 'royalblue'
                draw.line((endx1, endy1, endx2, endy2), fill=fill_c, width=1)
                
    return np.array(img)[...,::-1] # 保存的结果是rgb

def create_collide_bev(pred_pts_bbox, gt_pts_bbox, only_box_for_col_det=None, PIXELS_PER_METER = 5, size = (300,300)):
    img_box = Image.new('L', size)
    draw = ImageDraw.Draw(img_box)
    img_wp = Image.new('L', size)
    draw_wp = ImageDraw.Draw(img_wp)
    origin = (size[0]//2, size[1]//2)

    gt_bbox = gt_pts_bbox.get('boxes_3d', None)
    gt_bbox = None
    pred_bbox = pred_pts_bbox.get('boxes_3d', None)
    for box_type_idx, (bbox, c) in enumerate([(gt_bbox,"chartreuse"), (pred_bbox,"dodgerblue")]):
        if bbox is None:
            continue
        arrs = copy.deepcopy(bbox.tensor.detach().cpu().numpy())
        arrs[:,0] += 1.3

        for idx, arr in enumerate(arrs):#(9,)
            box = {}
            box['position']=(arr[0],arr[1],arr[2])
            box['extent']=(arr[3+2],arr[3+0],arr[3+1])
            box['yaw']=arr[6]
            box['speed']=np.sqrt(arr[7]**2+arr[8]**2)

            x = -box['position'][1]*PIXELS_PER_METER + origin[1]#y是向右
            y = -box['position'][0]*PIXELS_PER_METER + origin[0]#x是向前 本来就应该是-
            yaw = box['yaw']
            extent_x = box['extent'][2]*PIXELS_PER_METER/2
            extent_y = box['extent'][1]*PIXELS_PER_METER/2
            
            p1, p2, p3, p4 = get_coords_BB(x, y, yaw, extent_x, extent_y)
            draw.polygon((p1, p2, p3, p4), fill=1)
            
    front = only_box_for_col_det['front']
    width = only_box_for_col_det['width']
    forshow = only_box_for_col_det['forshow']
    wp = pred_pts_bbox.get('attrs_3d', None)
    if wp is None:
        return None
    wp = wp.clone()
    wp[:,0] += 1.3 # 变到ego系了
    wp[:,0] += front
    args = []
    for idp, p in enumerate([wp[0],wp[3]]): # 第0个点和最后一个点直线
        x = p[1]*PIXELS_PER_METER + origin[1]
        y = -p[0]*PIXELS_PER_METER + origin[0]
        args.append(x)
        args.append(y)
    draw_wp.line(args,fill=1,width=width*PIXELS_PER_METER)
    
    img_box = np.array(img_box)
    img_wp = np.array(img_wp)
    if forshow:
        img_save = np.stack([img_box*255, img_wp*255, np.zeros_like(img_box)], axis=-1)
        return img_save
    else:
        img_and = np.logical_and(img_box, img_wp)
        return img_and
    
def create_bev(pred_pts_bbox, gt_pts_bbox, PIXELS_PER_METER = 5, size = (300,300), wploss=None):
    if gt_pts_bbox['topdown'] is not None:
        img = Image.fromarray(mmcv.imread(gt_pts_bbox['topdown']))
        size = img.size
    else:
        img = Image.new('RGB', size)
    draw = ImageDraw.Draw(img)
    origin = (size[0]//2, size[1]//2)

    gt_bbox = gt_pts_bbox.get('boxes_3d', None)
    gt_wp_attn = gt_pts_bbox.get('wp_attn', None)
    gt_idxs = gt_pts_bbox.get('gt_idxs', None)
    matched_idxs = pred_pts_bbox.get('matched_idxs', None)
    pred_bbox = pred_pts_bbox.get('boxes_3d', None)
    wp_attn = pred_pts_bbox.get('wp_attn', None)
    
    if gt_wp_attn is not None:
        gt_wp_attn = gt_wp_attn.mean(0)
    if wp_attn is not None:
        wp_attn = wp_attn.mean(0)
    # if gt_wp_attn is not None:
    #     gt_wp_attn = gt_wp_attn[1]
    # if wp_attn is not None:
    #     wp_attn = wp_attn[1]
    
    for box_type_idx, (bbox, c) in enumerate([(gt_bbox,"chartreuse"), (pred_bbox,"dodgerblue")]):
        if bbox is None:
            continue
        arrs = copy.deepcopy(bbox.tensor.detach().cpu().numpy())
        arrs[:,0] += 1.3

        for idx, arr in enumerate(arrs):#(9,)
            box = {}
            box['position']=(arr[0],arr[1],arr[2])
            box['extent']=(arr[3+2],arr[3+0],arr[3+1])
            box['yaw']=arr[6]
            box['speed']=np.sqrt(arr[7]**2+arr[8]**2)

            x = -box['position'][1]*PIXELS_PER_METER + origin[1]#y是向右
            y = -box['position'][0]*PIXELS_PER_METER + origin[0]#x是向前 本来就应该是-
            yaw = box['yaw']
            extent_x = box['extent'][2]*PIXELS_PER_METER/2
            extent_y = box['extent'][1]*PIXELS_PER_METER/2
            
            p1, p2, p3, p4 = get_coords_BB(x, y, yaw, extent_x, extent_y)

            # point = (int(x),int(y))
            # draw.text(point, f"{np.array(arr)[:2].astype(int)}", fill='white', width=2) # center
            if box_type_idx==0:
                draw.polygon((p1, p2, p3, p4), fill=c, outline='black')
            else:
                draw.polygon((p1, p2, p3, p4), outline=c)

            if 'speed' in box:
                vel = box['speed']*3 #/3.6 # in m/s # just for visu
                endx1, endy1, endx2, endy2 = get_coords(x, y, yaw, vel)
                fill_c = 'orange' if box_type_idx==0 else 'royalblue'
                draw.line((endx1, endy1, endx2, endy2), fill=fill_c, width=1)
                
                if box_type_idx==1: # 预测
                    if wp_attn is not None:
                        draw.text((endx1+30, endy1), f"{wp_attn[idx+1]:.2f}", fill='blue')
                        # 第一个东西是对cls_emb的attn 错误
                        # 这个东西是和预测阈值的数量挂钩的，是过滤后的东西，倒是可以在前后拼接
                        # 这个+1没问题
                    if matched_idxs is not None:
                        draw.text((endx1+30+30, endy1), f"{int(matched_idxs[idx])}", fill='blue')
                else:
                    if gt_wp_attn is not None:
                        # gt_wp_attn比gt_idxs多一些，因为由对route的attn
                        # import pdb;pdb.set_trace()
                        draw.text((endx1+30+60, endy1), f"{gt_wp_attn[idx]:.2f}", fill='green') 
                        # 第一个东西是对cls_emb的attn，第一次是ego对应attn是ego，没问题
                    if gt_idxs is not None:
                        draw.text((endx1+30+90, endy1), f"{int(gt_idxs[idx])}", fill='green') 
    
    # import pdb;pdb.set_trace()
    # if len(gt_wp_attn) != len(wp_attn):
    #     # import pdb;pdb.set_trace()
    #     print(len(gt_wp_attn),len(wp_attn))
    if gt_wp_attn is not None and wp_attn is not None:
        assert len(gt_idxs) + 2 - 1 == len(gt_wp_attn) # 因为gt_idxs在可视化获取box的时候进行了前面补0，所以会多出一个
        str1 = ",".join([f"{t*100:.0f}" for t in gt_wp_attn])
        str1_ = "ego,"+','.join([f"{t:.0f}" for i,t in enumerate(gt_idxs) if i>0])+',route'
        str2 = ",".join([f"{t*100:.0f}" for t in wp_attn])
        str2_ = 'ego, ..., route1'
        draw.text((10,250), str1, fill='green') 
        draw.text((10,230), str1_, fill='green') 
        draw.text((10,270), str2, fill='blue') 
        draw.text((10,290), str2_, fill='blue') 
    
    gt_wp = copy.deepcopy(gt_pts_bbox.get('attrs_3d', None))
    pred_wp = copy.deepcopy(pred_pts_bbox.get('attrs_3d', None))
    # route_wp = copy.deepcopy(pred_pts_bbox.get('route_wp', None)) # 避免原地修改
    route_wp = None
    for idx, (wp, c) in enumerate([(gt_wp,"forestgreen"), (pred_wp,"blue"), (route_wp, 'orange')]):
        if wp is None:
            continue
        wp[:,0] += 1.3 # 从lidar到ego系
        for idp, p in enumerate(wp):
            x = p[1]*PIXELS_PER_METER + origin[1]
            y = -p[0]*PIXELS_PER_METER + origin[0]
            # 前面对box的处理，x和y都进行翻转，这里只是x进行翻转
            # 在fv可视化点的时候，根据可视化需要进行y方向的翻转
            draw.ellipse(((x-0.5,y-0.5),(x+0.5,y+0.5)), fill=c)
            draw.text((0,60+idx*50+idp*10), f"({p[0]:.1f},{p[1]:.1f})",fill=c)

    route = gt_pts_bbox.get('route', None)
    tp = gt_pts_bbox.get('tp', None)
    light = gt_pts_bbox.get('light', None)
    command = gt_pts_bbox.get('command', None)
    iscollide = pred_pts_bbox.get('iscollide', None)
    if route is not None:
        arr = route[0] # 第一个route，6d
        box = {}
        box['position']=(arr[0],arr[1])
        box['extent']=(arr[5],arr[4])
        box['yaw']=arr[2]

        x = -box['position'][1]*PIXELS_PER_METER + origin[1]#y是向右
        y = -box['position'][0]*PIXELS_PER_METER + origin[0]#x是向前 本来就应该是-
        # yaw = -box['yaw'] / 180 * np.pi
        yaw = box['yaw']
        extent_x = box['extent'][1]*PIXELS_PER_METER/2
        extent_y = box['extent'][0]*PIXELS_PER_METER/2

        # point = (int(x),int(y))
        # draw.text(point, f"{np.array(arr)[:3].astype(int)}", fill='white', width=2) # center
        
        p1, p2, p3, p4 = get_coords_BB(x, y, yaw, extent_x, extent_y)
        
        draw.polygon((p1, p2, p3, p4), outline='coral')

    if tp is not None:
        x = tp[1]*PIXELS_PER_METER + origin[1]
        y = -tp[0]*PIXELS_PER_METER + origin[0]  
        point = (int(x), int(y))
        draw.ellipse(((point[0]-1,point[1]-1),(point[0]+1,point[1]+1)), fill="pink", width=3)
        draw.text((0,0), f"target:{np.array(tp)[:2].astype(int)}",fill='pink',width=2)
    
    if light is not None:
        c = 'red' if light==1 else 'green'
        draw.text((0,10), f"light",fill=c,width=2)
    
    if command is not None:
        if isinstance(command, np.ndarray):
            command = command.item()
        draw.text((0,20), f"cmd:{command},{num2str[command]}", fill='peru', width=2)

    if iscollide is not None:
        fill_ = 'red' if iscollide else 'green'
        string = 'true' if iscollide else 'false'
        draw.text((0,30), f"iscollide: {string}", fill=fill_, width=2)

    if wploss is not None:
        draw.text((0,40), f"wploss: {wploss.item():.2f}", fill='red', width=2)
    img = np.array(img)[:,:,::-1]
    return img

def get_coords(x, y, yaw, vel):
    length = -vel
    endx2 = x + length * math.sin(yaw)
    endy2 = y + length * math.cos(yaw)

    return x, y, endx2, endy2  

def get_coords_BB(x, y, yaw, extent_x, extent_y):
    endx1 = x - extent_x * math.cos(yaw) - extent_y * math.sin(yaw)
    endy1 = y + extent_x * math.sin(yaw) - extent_y * math.cos(yaw)

    endx2 = x + extent_x * math.cos(yaw) - extent_y * math.sin(yaw)
    endy2 = y - extent_x * math.sin(yaw) - extent_y * math.cos(yaw)

    endx3 = x + extent_x * math.cos(yaw) + extent_y * math.sin(yaw)
    endy3 = y - extent_x * math.sin(yaw) + extent_y * math.cos(yaw)

    endx4 = x - extent_x * math.cos(yaw) + extent_y * math.sin(yaw)
    endy4 = y + extent_x * math.sin(yaw) + extent_y * math.cos(yaw)

    return (endx1, endy1), (endx2, endy2), (endx3, endy3), (endx4, endy4)