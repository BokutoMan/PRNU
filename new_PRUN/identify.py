import numpy as np
import pickle
from scipy.stats import pearsonr
import pywt

# 小波变换去噪
def wavelet_denoise(image):
    coeffs = pywt.wavedec2(image, 'db1', level=1)
    cA, (cH, cV, cD) = coeffs
    cH.fill(0)
    cV.fill(0)
    cD.fill(0)
    denoised_image = pywt.waverec2((cA, (cH, cV, cD)), 'db1')
    return denoised_image

# 计算PRNU指纹之间的皮尔逊相关系数来进行PRNU的模式匹配
def compute_correlation(prnu1, prnu2):
    # 将PRNU展平为一维数组
    prnu1_flat = prnu1.flatten()
    prnu2_flat = prnu2.flatten()
    
    # 使用Pearson相关系数计算两个PRNU之间的相关性
    correlation, _ = pearsonr(prnu1_flat, prnu2_flat)
    return correlation

# 对图片进行检验，分析相机源
def identify_image(prnu_database, test_image):
    """
    通过比较测试图像的PRNU与数据库中的PRNU来识别图像的相机型号。

    参数:
    prnu_database (dict): PRNU数据库，其中键是相机型号，值是相应的PRNU。
    test_image (numpy.ndarray): 测试图像。

    返回:
    tuple: 识别出的相机型号和与数据库中最高相关性的相关系数。
    """

    # 对测试图像进行小波去噪，计算噪声残差，并提取PRNU
    test_denoised_image = wavelet_denoise(test_image)
    test_noise_residual = test_image - test_denoised_image
    test_prnu = (test_noise_residual - np.mean(test_noise_residual)) / np.std(test_noise_residual)

    best_match = None  # 最佳匹配的相机型号
    highest_correlation = -1  # 最高相关性
    for camera, prnu in prnu_database.items():
        # 计算测试PRNU与数据库中PRNU的相关性
        correlation = compute_correlation(test_prnu, prnu)
        
        # 更新最高相关性和最佳匹配的相机型号
        if correlation > highest_correlation:
            highest_correlation = correlation
            best_match = camera
    
    return best_match, highest_correlation  # 返回最佳匹配的相机型号和相关系数


# PRNU的匹配和验证，计算该方法的识别acc
def evaluate_prnu(prnu_database, test_images_with_camera):
    """
    评估PRNU识别系统的性能。

    参数:
    prnu_database (dict): PRNU数据库，其中键是相机型号，值是相应的PRNU。
    test_images_with_camera (dict): 测试图像的字典，其中键是相机型号，值是相应的测试图像列表。

    返回:
    float: PRNU识别系统的准确率。
    """

    correct = 0  # 正确识别的图像数量
    total = 0  # 总共测试的图像数量

    # 遍历每个相机的测试图像
    for camera, test_images in test_images_with_camera.items():
        camera_correct = 0
        for test_image in test_images:
            # 识别测试图像的相机型号，并获取相关性
            identified_camera, correlation = identify_image(prnu_database, test_image)
            # 解除注释可观察进度
            # print(f"测试图像识别为: {identified_camera} (实际: {camera})，相关性: {correlation:.4f}", end='\r')

            # 检查识别结果是否正确
            if identified_camera == camera:
                correct += 1
                camera_correct += 1
            total += 1
            print()
        print(f"{camera} 识别准确率: {camera_correct/len(test_images)}")
    # 计算准确率
    accuracy = correct / total if total > 0 else 0
    return accuracy  # 返回准确率


if __name__ == "__main__":
    # 定义目标图像大小
    target_size = (1024 * 2, 1536 * 2)

    # 加载PRNU数据库和测试图像字典
    prnu_database = pickle.load(open('prnu_database.pkl', 'rb'))
    test_images_with_camera = pickle.load(open('test_images_with_camera.pkl', 'rb'))

    # 评估PRNU识别系统的准确率
    accuracy = evaluate_prnu(prnu_database, test_images_with_camera)

    # 打印识别准确率
    print(f"识别准确率: {accuracy:.4f}")