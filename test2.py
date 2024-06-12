import numpy as np
import cv2
import pywt
import os

def preprocess_image(image):
    # 归一化处理
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized_image

def denoise_wavelet(image, wavelet='db1', level=1):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    # 将细节系数进行阈值处理
    threshold = np.sqrt(2 * np.log(image.size))
    new_coeffs = list(coeffs)
    new_coeffs[1:] = [pywt.threshold(detail, threshold, mode='soft') for detail in coeffs[1:]]  # 正确处理细节系数
    denoised_image = pywt.waverec2(new_coeffs, wavelet)
    return denoised_image

def extract_prnu(images):
    prnu_accumulator = np.zeros_like(images[0], dtype=np.float32)
    
    for image in images:
        preprocessed = preprocess_image(image)
        denoised = denoise_wavelet(preprocessed)
        residual = preprocessed - denoised
        prnu_accumulator += residual
    
    prnu_pattern = prnu_accumulator / len(images)
    return prnu_pattern

def save_prnu(prnu_pattern, device_id, output_dir='prnu_library'):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'{device_id}_prnu.npy')
    np.save(filename, prnu_pattern)


# 示例使用
device_id = 'camera_001'
image_files = ['1.pgm']  # 替换为实际文件路径
images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in image_files]

prnu_pattern = extract_prnu(images)
save_prnu(prnu_pattern, device_id)
