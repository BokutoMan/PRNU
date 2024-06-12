import os
import random
import rawpy
import cv2
import numpy as np
import pywt
import tqdm
import pickle
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 以灰度形式读取RAW图像
def preprocess_image(image_path, crop_percent=0.05, target_size=(512, 512)):
    with rawpy.imread(image_path) as raw:
        rgb_image = raw.postprocess()
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    # 裁剪图像边缘
    h, w = gray_image.shape
    crop_h, crop_w = int(h * crop_percent), int(w * crop_percent)
    cropped_image = gray_image[crop_h:h-crop_h, crop_w:w-crop_w]
    
    # 调整大小
    resized_image = cv2.resize(cropped_image, target_size)
    return resized_image

# 小波变换去噪
def wavelet_denoise(image):
    coeffs = pywt.wavedec2(image, 'db1', level=1)
    cA, (cH, cV, cD) = coeffs
    cH.fill(0)
    cV.fill(0)
    cD.fill(0)
    denoised_image = pywt.waverec2((cA, (cH, cV, cD)), 'db1')
    return denoised_image

# 提取PRNU
def extract_prnu(images):
    prnu = np.zeros_like(images[0], dtype=np.float32)
    for image in tqdm.tqdm(images, desc="Extracting PRNU", unit="image"):
        denoised_image = wavelet_denoise(image)
        noise_residual = image - denoised_image
        prnu += noise_residual
    prnu /= len(images)
    prnu = (prnu - np.mean(prnu)) / np.std(prnu)
    return prnu

# 载入图像数据
def load_images_from_folder(folder, test_ratio=0.2, target_size=(512, 512)):
    images = []
    test_images = []
    filenames = [f for f in os.listdir(folder) if f.lower().endswith(('.cr2', '.dng', '.nef', '.pef'))]
    random.shuffle(filenames)
    split_index = int(len(filenames) * (1 - test_ratio))
    
    for i, filename in enumerate(filenames):
        img_path = os.path.join(folder, filename)
        print(f"Loading image: {img_path}", end='\r')
        img = preprocess_image(img_path, target_size=target_size)
        if img is not None:
            if i < split_index:
                images.append(img)
            else:
                test_images.append(img)
    print()
    return images, test_images

# 构建PRNU库
def build_prnu_database(base_path, test_ratio=0.2, target_size=(512, 512)):
    prnu_database = {}
    test_images_by_camera = {}
    
    for camera_folder in os.listdir(base_path):
        camera_path = os.path.join(base_path, camera_folder)
        if os.path.isdir(camera_path):
            images, test_images = load_images_from_folder(camera_path, test_ratio, target_size)
            prnu = extract_prnu(images)
            prnu_database[camera_folder] = prnu
            test_images_by_camera[camera_folder] = test_images
    
    return prnu_database, test_images_by_camera

# 主程序
def main(base_path, test_ratio, target_size):
    prnu_database, test_images_by_camera = build_prnu_database(base_path, test_ratio, target_size)

    with open('prnu_database.pkl', 'wb') as f:
        pickle.dump(prnu_database, f)
    with open('test_images_by_camera.pkl', 'wb') as f:
        pickle.dump(test_images_by_camera, f)

    # 显示第一张图片去噪前后的对比
    first_image = None
    for camera_folder, images in test_images_by_camera.items():
        if images:
            first_image = images[0]
            break

    if first_image is not None:
        denoised_image = wavelet_denoise(first_image)
        
        # 显示原始图像和去噪后的图像
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(first_image, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title('Denoised Image')
        plt.imshow(denoised_image, cmap='gray')
        plt.show()

    prnu_database = pickle.load(open('prnu_database.pkl', 'rb'))
    test_images_by_camera = pickle.load(open('test_images_by_camera.pkl', 'rb'))

    print(prnu_database)

if __name__ == "__main__":
    base_path = './BOSSbase'
    test_ratio=0.2
    target_size=(2048, 3072)
    main(base_path,test_ratio,target_size)
