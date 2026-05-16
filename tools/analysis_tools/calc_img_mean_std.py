import os
import glob
import cv2
import numpy as np
from tqdm import tqdm   # 进度条，若未安装可注释掉

def compute_mean_std(image_paths):
    """
    计算图像数据集的每个通道的均值和标准差（总体标准差）
    Args:
        image_paths: 图像路径列表
    Returns:
        mean: 各通道均值 (BGR顺序)
        std:  各通道标准差 (BGR顺序)
    """
    sum_pixels = np.zeros(3, dtype=np.float64)   # 各通道像素累加和
    sum_sq_pixels = np.zeros(3, dtype=np.float64) # 各通道像素平方累加和
    total_pixels = 0

    for path in tqdm(image_paths, desc="Processing"):
        img = cv2.imread(path)   # 读取为 BGR 顺序，shape=(H,W,3)
        if img is None:
            print(f"Warning: cannot read {path}")
            continue
        img = img.astype(np.float64)  # 避免溢出
        sum_pixels += img.sum(axis=(0, 1))
        sum_sq_pixels += (img ** 2).sum(axis=(0, 1))
        total_pixels += img.shape[0] * img.shape[1]

    mean = sum_pixels / total_pixels
    std = np.sqrt(sum_sq_pixels / total_pixels - mean ** 2)
    return mean, std

def main():
    # 修改为您的实际路径
    train_dir = "/home/vipuser/project/mmdetection3d/data/view_of_delft_PUBLIC/radar_3frames/training/image_2"
    # test_dir  = "/home/vipuser/project/mmdetection3d/data/view_of_delft_PUBLIC/radar_3frames/testing/image_2"
    test_dir  = "/home/vipuser/project/mmdetection3d/data/view_of_delft_PUBLIC/radar_3frames/testing"

    # 收集所有 .jpg 文件
    train_paths = glob.glob(os.path.join(train_dir, "*.jpg"))
    test_paths  = glob.glob(os.path.join(test_dir, "*.jpg"))
    all_paths = train_paths + test_paths
    print(f"Training images: {len(train_paths)}, Testing images: {len(test_paths)}, Total: {len(all_paths)}")

    if not all_paths:
        print("No images found. Check directory paths.")
        return

    # 计算统计量 (BGR 顺序)
    mean_bgr, std_bgr = compute_mean_std(all_paths)

    # 输出 BGR 顺序结果
    print("\n=== BGR order (as read by OpenCV) ===")
    print(f"mean = [{mean_bgr[0]:.3f}, {mean_bgr[1]:.3f}, {mean_bgr[2]:.3f}]")
    print(f"std  = [{std_bgr[0]:.3f}, {std_bgr[1]:.3f}, {std_bgr[2]:.3f}]")

    # 转换为 RGB 顺序（方便对比）
    mean_rgb = mean_bgr[::-1]
    std_rgb  = std_bgr[::-1]
    print("\n=== RGB order (converted) ===")
    print(f"mean = [{mean_rgb[0]:.3f}, {mean_rgb[1]:.3f}, {mean_rgb[2]:.3f}]")
    print(f"std  = [{std_rgb[0]:.3f}, {std_rgb[1]:.3f}, {std_rgb[2]:.3f}]")

if __name__ == "__main__":
    main()