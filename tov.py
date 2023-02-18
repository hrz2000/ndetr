import os  # python标准库，不需要安装，用于系统文件操作相关
import cv2  # python非标准库，pip install opencv-python 多媒体处理
from PIL import Image  # python非标准库，pip install pillow，图像处理
import moviepy.editor as mov  # python非标准库，pip install moviepy，多媒体编辑
from glob import glob
import numpy as np
import argparse
from tqdm import tqdm

def image_to_video(image_path, media_path):
    '''
    图片合成视频函数
    :param image_path: 图片路径
    :param media_path: 合成视频保存路径
    :return:
    '''
    # 获取图片路径下面的所有图片名称
    image_names = glob(image_path)
    print(f"imgs num: {len(image_names)}")
    # 对提取到的图片名称进行排序
    image_names.sort()
    # 设置写入格式
    # fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # 设置每秒帧数
    fps = 5  # 由于图片数目较少，这里设置的帧数比较低
    # 读取第一个图片获取大小尺寸，因为需要转换成视频的图片大小尺寸是一样的
    image = Image.open(image_names[0])
    # 初始化媒体写入对象
    media_writer = cv2.VideoWriter(media_path, fourcc, fps, image.size)
    # 遍历图片，将每张图片加入视频当中
    for image_name in tqdm(image_names):
        im = Image.open(image_name)
        # print(np.array(im).shape)
        media_writer.write(np.array(im))
        # print(image_name, '合并完成！')
    # 释放媒体写入对象
    media_writer.release()
    print('视频写入完成！')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--re', '-r', default="output/output/detr_eval_split/2023-02-17_10:55:27/Route_8/longest_weathers_8_route0_02_17_11_16_45_2023-02-17_11:17:02/vis/*", type=str)
    parser.add_argument('--save', '-s', default='8灯.mp4', type=str)
    arg = parser.parse_args()
    image_to_video(arg.re, arg.save)
    print("ok")