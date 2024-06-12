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

def preprocess_image(image_path, crop_percent=0.05, target_size=(512, 512)):
    """
    以灰度形式读取RAW图像，并进行预处理。

    参数:
    image_path (str): 图像文件的路径。
    crop_percent (float): 裁剪边缘的百分比（默认为 0.05）。
    target_size (tuple): 图像的目标尺寸（默认为 (512, 512)）。

    返回:
    numpy.ndarray: 预处理后的灰度图像。
    """

    # 使用 rawpy 库以RAW格式读取图像
    with rawpy.imread(image_path) as raw:
        # 将RAW图像后处理为RGB图像
        rgb_image = raw.postprocess()

    # 将RGB图像转换为灰度图像
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    # 裁剪图像边缘
    h, w = gray_image.shape
    crop_h, crop_w = int(h * crop_percent), int(w * crop_percent)
    cropped_image = gray_image[crop_h:h-crop_h, crop_w:w-crop_w]
    
    # 调整图像大小
    resized_image = cv2.resize(cropped_image, target_size)
    return resized_image  # 返回预处理后的图像

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
    """
    提取图像集合的噪声残差噪声模式（PRNU）。

    参数:
    images (list): 图像集合。

    返回:
    numpy.ndarray: 提取的PRNU。
    """

    prnu = np.zeros_like(images[0], dtype=np.float32)  # 创建与第一张图像相同大小的全零数组作为初始PRNU

    # 遍历图像集合，逐张提取PRNU
    for image in tqdm.tqdm(images, desc="提取PRNU", unit="张"):
        # 对图像进行小波去噪
        denoised_image = wavelet_denoise(image)
        
        # 计算噪声残差
        noise_residual = image - denoised_image
        
        # 将噪声残差叠加到PRNU中
        prnu += noise_residual

    # 对PRNU取平均
    prnu /= len(images)
    
    # 将PRNU标准化（均值为0，标准差为1）
    prnu = (prnu - np.mean(prnu)) / np.std(prnu)
    
    return prnu  # 返回提取的PRNU

# 载入图像数据
def load_images_from_folder(folder, test_ratio=0.2, target_size=(512, 512)):
    """
    从文件夹加载图片，并将其分为训练集和测试集。

    参数:
    folder (str): 包含图片的文件夹路径。
    test_ratio (float): 用于测试的图片比例（默认为 0.2）。
    target_size (tuple): 图片的目标尺寸（默认为 (512, 512)）。

    返回:
    tuple: 包含两个图片列表的元组：训练图片和测试图片。
    """

    images = []  # 存储训练图片的列表
    test_images = []  # 存储测试图片的列表

    # 获取文件夹中图片的文件名列表
    filenames = [f for f in os.listdir(folder) if f.lower().endswith(('.cr2', '.dng', '.nef', '.pef'))]

    # 随机打乱文件名列表
    random.shuffle(filenames)

    # 计算将数据拆分为训练集和测试集的索引位置
    split_index = int(len(filenames) * (1 - test_ratio))

    # 遍历文件名列表，加载每个图片
    for i, filename in enumerate(filenames):
        # 构建完整的图片文件路径
        img_path = os.path.join(folder, filename)

        # 打印加载图片的进度
        print(f"Loading image: {img_path}", end='\r')

        # 预处理图片（例如，调整大小）
        img = preprocess_image(img_path, target_size=target_size)

        # 检查图片是否成功加载
        if img is not None:
            # 根据 split_index 将图片分为训练集和测试集
            if i < split_index:
                images.append(img)  # 将图片添加到训练集
            else:
                test_images.append(img)  # 将图片添加到测试集

    print()  # 加载图片完成后打印一个新行
    return images, test_images  # 返回训练集和测试集

# 构建PRNU库的函数
def build_prnu_database(base_path, test_ratio=0.2, target_size=(512, 512)):
    # 初始化PRNU数据库和测试图像字典
    prnu_database = {}
    test_images_by_camera = {}
    
    # 遍历相机文件夹
    for camera_folder in os.listdir(base_path):
        # 获取相机文件夹的完整路径
        camera_path = os.path.join(base_path, camera_folder)
        
        # 检查是否为目录
        if os.path.isdir(camera_path):
            # 加载相机文件夹中的图像数据
            images, test_images = load_images_from_folder(camera_path, test_ratio, target_size)
            
            # 提取训练图像的PRNU
            prnu = extract_prnu(images)
            
            # 将PRNU和测试图像添加到相应的字典中
            prnu_database[camera_folder] = prnu
            test_images_by_camera[camera_folder] = test_images
    
    # 返回构建好的PRNU数据库和测试图像字典
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
    base_path = "D:\Download\BaiduPan\BossBase"
    test_ratio=0.2
    target_size=(2048, 3072)
    main(base_path,test_ratio,target_size)
