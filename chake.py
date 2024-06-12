import numpy as np
import cv2
import os
import pywt

# 图像预处理函数
def preprocess_image(image):
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized_image

# 小波去噪函数
def denoise_wavelet(image, wavelet='db1', level=1):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    threshold = np.sqrt(2 * np.log(image.size))
    denoised_coeffs = [coeffs[0]]
    denoised_coeffs += [pywt.threshold(detail, threshold, mode='soft') for detail in coeffs[1:]]
    denoised_image = pywt.waverec2(denoised_coeffs, wavelet)
    return denoised_image

# 提取PRNU模式函数
def extract_prnu(images, wavelet='db1', level=1):
    prnu_accumulator = np.zeros_like(images[0], dtype=np.float32)
    for image in images:
        preprocessed = preprocess_image(image)
        denoised_image = denoise_wavelet(preprocessed, wavelet=wavelet, level=level)
        residual = preprocessed - denoised_image
        prnu_accumulator += residual
    prnu_pattern = prnu_accumulator / len(images)
    return prnu_pattern

# 保存PRNU模式函数
def save_prnu(prnu_pattern, device_id, output_dir='prnu_library'):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'{device_id}_prnu.npy')
    np.save(filename, prnu_pattern)

# 加载PRNU模式函数
def load_prnu(device_id, input_dir='prnu_library'):
    filename = os.path.join(input_dir, f'{device_id}_prnu.npy')
    prnu_pattern = np.load(filename)
    return prnu_pattern

# 计算皮尔逊相关系数
def calculate_pearson_correlation(image_prnu, prnu_library):
    correlation_scores = []
    for prnu_pattern in prnu_library:
        # 对信号进行零均值化
        image_prnu_zero_mean = image_prnu - np.mean(image_prnu)
        prnu_pattern_zero_mean = prnu_pattern - np.mean(prnu_pattern)
        # 计算皮尔逊相关系数
        correlation = np.corrcoef(image_prnu_zero_mean.flatten(), prnu_pattern_zero_mean.flatten())[0, 1]
        correlation_scores.append(correlation)
    return correlation_scores

# 示例使用
device_id = 'camera_001'
image_files = ['image1.jpg', 'image2.jpg', 'image3.jpg']  # 替换为实际文件路径
images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in image_files]
prnu_pattern = extract_prnu(images, wavelet='db1', level=1)
save_prnu(prnu_pattern, device_id)

# 加载PRNU库
prnu_library = []
for device_id in ['camera_001', 'camera_002', 'camera_003']:  # 替换为实际设备ID列表
    prnu_library.append(load_prnu(device_id))

# 计算待检测图像的PRNU模式
test_image_files = ['test_image.jpg']  # 替换为实际测试图像文件路径
test_images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in test_image_files]
test_prnu_pattern = extract_prnu(test_images, wavelet='db1', level=1)

# 计算相关性
correlation_scores = calculate_pearson_correlation(test_prnu_pattern, prnu_library)
print("Correlation scores:", correlation_scores)

# 判断相关性
threshold = 0.5  # 设定一个相关性阈值，具体值需根据实验结果调整
matching_device_id = None
for idx, score in enumerate(correlation_scores):
    if score > threshold:
        matching_device_id = ['camera_001', 'camera_002', 'camera_003'][idx]
        break

if matching_device_id:
    print(f"The test image matches with device {matching_device_id}")
else:
    print("No matching device found")
