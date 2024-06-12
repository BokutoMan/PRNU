import numpy as np
import cv2
import os
import pywt

# 图像预处理函数，归一化处理
def preprocess_image(image):
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized_image

# 小波去噪函数
def denoise_wavelet(image, wavelet='db1', level=1):
    # 对输入图像进行小波变换，得到其小波系数
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    # 计算软阈值的阈值
    threshold = np.sqrt(2 * np.log(image.size))
    # 对每个通道的小波系数进行软阈值处理
    denoised_coeffs = [pywt.threshold(detail, threshold, mode='soft') for detail in coeffs]
    denoised_coeffs[1] = (denoised_coeffs[1][0],denoised_coeffs[1][1],denoised_coeffs[1][2])
    return denoised_coeffs  # 返回处理后的小波系数

# 提取PRNU模式的函数
def extract_prnu(images, wavelet='db1', level=1):
    # 初始化PRNU累加器
    prnu_accumulator = np.zeros_like(images[0], dtype=np.float32)
    for image in images:
        # 对输入图像进行预处理，如归一化
        preprocessed = preprocess_image(image)
        # 对预处理后的图像进行小波去噪，得到小波系数
        denoised_coeffs = denoise_wavelet(image, wavelet, level=level)
        # 使用小波系数重构图像，得到去噪后的图像
        denoised_image = pywt.waverec2(denoised_coeffs, wavelet=wavelet)
        # 计算残差，即原始图像减去去噪后的图像
        residual = preprocessed - denoised_image
        # 将残差累加到PRNU累加器中
        prnu_accumulator += residual
    # 计算PRNU模式，即PRNU累加器除以图像数量
    prnu_pattern = prnu_accumulator / len(images)
    return prnu_pattern  # 返回计算得到的PRNU模式

def save_prnu(prnu_pattern, device_id, output_dir='prnu_library'):
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


