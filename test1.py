import numpy as np
import cv2
import os

import pywt
from use1 import extract_prnu,devices,first_num,img_path
from math import sqrt
# 加载已知设备的PRNU噪声模式库
paths = [f'prnu_library\camera_00{i}_prnu.npy' for i in range(1,7)]
prnu_library = []  # 已经保存的设备PRNU噪声模式
for path in paths:
    prnu_library.append(np.load(path))


def get_pc(x,y):
    n=len(x)
    x_mean = x.sum()/n
    y_mean = y.sum()/n
    x = x - x_mean
    y = y - y_mean
    fz = (x*y).sum()
    fm = (x*x).sum() * (y*y).sum()
    fm = sqrt(fm)

    return fz/fm


def calculate_pearson_correlation(image_prnu, prnu_library):
    correlation_scores = []
    img_prnu = extract_prnu([image_prnu])
    for prnu_pattern in prnu_library:
        pc = get_pc(img_prnu,prnu_pattern)
        correlation_scores.append(pc)
    return correlation_scores

if __name__ == "__main__":
    # 提取待匹配图像的PRNU噪声模式
    for k in range(0,6):
        image_files =  [os.path.join(img_path, f"{i}.pgm") for i in range(first_num[k] + 501, first_num[k] + 701)]  # 使用1-500提取第一个型号
        images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in image_files]
        num_ok = 0
        i = 0
        best_index = {
            1:0,
            2:0,
            3:0,
            4:0,
            5:0,
            6:0
        }
        for image_prnu in images:
            i += 1
            # 找到最佳匹配的设备
            correlation_scores = calculate_pearson_correlation(image_prnu,prnu_library)
            best_match_index = np.argmax(correlation_scores)
            best_index[best_match_index+1] += 1
            # print(f"NO.{i}: ",best_match_index, correlation_scores)
            
            if best_match_index == k:
                num_ok += 1
        print(f" for {devices[k]}, the right number is {num_ok}, acc = {num_ok/400}", best_index)

