import numpy as np
import cv2
import pywt
import os

def preprocess_image(image):
    """预处理图像：归一化处理"""
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized_image

def denoise_wavelet(image, wavelet='db1', level=1):
    """小波去噪"""
    # 小波变换：将图像分解成多个尺度
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    # 阈值选择：根据噪声水平选择阈值
    threshold = np.sqrt(2 * np.log(image.size))
    # 系数阈值处理：对细节系数进行软阈值处理
    denoised_coeffs = [coeffs[0]]  # 近似系数保持不变
    denoised_coeffs += [pywt.threshold(detail, threshold, mode='soft') for detail in coeffs[1:]]
    # 小波逆变换：重构去噪图像
    denoised_coeffs[1] = (denoised_coeffs[1][0],denoised_coeffs[1][1],denoised_coeffs[1][2])
    denoised_image = pywt.waverec2(denoised_coeffs, wavelet)
    return denoised_image

def extract_prnu(images, wavelet='db1', level=1):
    """提取PRNU模式"""
    prnu_accumulator = np.zeros_like(images[0], dtype=np.float32)
    for image in images:
        preprocessed = preprocess_image(image)
        denoised_image = denoise_wavelet(preprocessed, wavelet=wavelet, level=level)
        residual = preprocessed - denoised_image
        prnu_accumulator += residual
    prnu_pattern = prnu_accumulator / len(images)
    return prnu_pattern

def save_prnu(prnu_pattern, device_id, output_dir='prnu_library'):
    """保存PRNU模式"""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'{device_id}_prnu.npy')
    np.save(filename, prnu_pattern)


devices = [
    "Canon EOS 400D",
    "Canon EOS 7DCanon",
    "EOS DIGITAL REBEL XSi",
    "PENTAX K20D",
    "NIKON D70",
    "M9 Digital Camera"
]
first_num = [
    1,
    1416,
    2770,
    5001,
    6210,
    8001
]
img_path = "D:\Download\database\BOSSbase_1.01"

if __name__ == "__main__":
    for k in range(0,6):
        device_id = f'camera_00{k+1}'
        image_files =  [os.path.join(img_path, f"{i}.pgm") for i in range(first_num[k], first_num[k] + 500)]  # 使用1-500提取第一个型号
        # print(image_files)
        images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in image_files]

        prnu_pattern = extract_prnu(images)
        save_prnu(prnu_pattern, device_id)
        print(devices[k],"saved")
