import os
import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage import gaussian_filter

# from skimage.metrics import peak_signal_noise_ratio as psnr1
# from skimage.metrics import structural_similarity as ssim1
import csv

def PSNR(img1, img2, maxI=255):
    # img1为参考图像 img2为待评估图像
    mse = np.mean((img1 - img2)**2)
    # print(f"mse: {mse:.2f}")
    psnr = 10 * np.log10(maxI*maxI/mse)
    # print(f"信噪比（PSNR）: {psnr:.2f} dB")
    return psnr

def SSIM(img1, img2, K1=0.01, K2=0.03, win_size=9, sigma=1.5, L=255):
    # img1为参考图像 img2为待评估图像
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    # 高斯加权滑动窗口
    mu1 = gaussian_filter(img1, sigma)
    mu2 = gaussian_filter(img2, sigma)
    # 方差 和 协方差
    NP = win_size ** 2
    cov_norm = NP / (NP - 1)  # 样本方差
    sigma1 = (gaussian_filter(img1 * img1, sigma) - mu1*mu1) * cov_norm
    sigma2 = (gaussian_filter(img2 * img2, sigma) -mu2*mu2) * cov_norm
    sigma12 = (gaussian_filter(img1 * img2, sigma) - mu1*mu2) * cov_norm
    # ssim
    A1 = 2*mu1*mu2+C1
    A2 = 2*sigma12+C2
    B1 = mu1**2 + mu2**2 + C1
    B2 = sigma1 + sigma2 + C2
    ssim = (A1*A2)/(B1*B2)
    ssim = np.mean(ssim)

    return ssim


# 保存文件
def save_scores(image_name, psnr_value, ssim_value, output_file):
    with open(output_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([image_name, psnr_value, ssim_value])

def main():
    original_images_dir = 'images/ref_img'  # 原始图像文件夹路径
    degraded_images_dir = 'images/dist_img'  # 退化图像文件夹路径
    output_csv = 'score_test.csv'  # 输出CSV文件路径
    # 确保输出CSV文件是空的或新建一个
    open(output_csv, 'w').close()
    # 列出原始图像文件夹中的所有图像
    cnt = 0
    for ref_file in os.listdir(original_images_dir):
        # print(ref_file)
        # 获得ref_img图片路径
        if ref_file.endswith('.png'):
            original_image_path = os.path.join(original_images_dir, ref_file)

        for filename in os.listdir(degraded_images_dir):
            cnt += 1
            # 获得dist_img图片路径
            if filename.endswith('.png'):
                degraded_image_path = os.path.join(degraded_images_dir, filename)
                # image = cv2.imread('images/dist_img/1600.AWGN.1.png', cv2.IMREAD_GRAYSCALE)
                # 检查图像是否成功读取
                # if image is None:
                #     print("Error: Image not found or cannot be read.")
                # else:
                #     # 显示图像
                #     cv2.imshow('Image', image)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
                #
                original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
                degraded_image = cv2.imread(degraded_image_path, cv2.IMREAD_GRAYSCALE)
                original_image = original_image.astype(np.float32)
                degraded_image = degraded_image.astype(np.float32)

                # 计算PSNR和SSIM
                psnr = PSNR(original_image, degraded_image)
                ssim = SSIM(original_image, degraded_image)
                # psnr_value, ssim_value = calculate_psnr_and_ssim(original_image, degraded_image)
                # aa = psnr1(original_image, degraded_image, data_range=255)
                # bb = ssim1(original_image, degraded_image, data_range=255, gaussian_weights=True)
                print(cnt, ref_file, filename, psnr, ssim)
                # 保存分数
                save_scores(filename, psnr, ssim, output_csv)

            if cnt%5==0:
                break


if __name__ == "__main__":
    main()
    print("calculate ending----------------")
