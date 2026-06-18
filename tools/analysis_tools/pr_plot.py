#!/usr/bin/env python3
"""
Plot mAP curves (average precision over all classes) from multiple pr_details.pkl files.
Supports Chinese labels and titles.

Usage example:
    python pr_plot.py --files results1/pr_details.pkl results2/pr_details.pkl \
                      --labels "模型A" "模型B" \
                      --eval_type 3d --diff_idx 0 --output map_curves.png
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ==================== 强制设置中文字体（直接指定文件路径） ====================
font_paths = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",   # 文泉驿微米黑
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",     # 文泉驿正黑
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # 思源黑体
    "/System/Library/Fonts/PingFang.ttc",               # macOS 苹方
    "C:/Windows/Fonts/msyh.ttc",                        # Windows 微软雅黑
]
chinese_font = None
for path in font_paths:
    if os.path.exists(path):
        chinese_font = fm.FontProperties(fname=path)
        fm.fontManager.addfont(path)
        plt.rcParams['font.family'] = chinese_font.get_name()
        print(f"成功加载中文字体：{path}")
        break

if chinese_font is None:
    print("警告：未找到预设的中文字体文件，将尝试使用系统字体名称...")
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
else:
    plt.rcParams['axes.unicode_minus'] = False

# ========== 设置全局字体大小（增大所有文字） ==========
plt.rcParams['font.size'] = 14          # 全局字体基准大小
plt.rcParams['axes.labelsize'] = 15     # 坐标轴标签大小
plt.rcParams['axes.titlesize'] = 16     # 子图标题大小（主标题用 title 参数可单独设置）
plt.rcParams['legend.fontsize'] = 12    # 图例字体大小
plt.rcParams['xtick.labelsize'] = 12    # X轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12    # Y轴刻度字体大小
# =================================================

def compute_mean_precision(pr_details, eval_type='3d', diff_idx=0, overlap_idx=1):
    """
    Compute average precision across all classes for a given evaluation type and difficulty index.

    Args:
        pr_details (dict): Loaded PR detail dict, e.g. {'3d': ndarray, ...}
        eval_type (str): '3d', 'bev', or 'bbox'
        diff_idx (int): Difficulty index (0 for first difficulty level)
        overlap_idx (int): 0 for strict (higher IoU), 1 for lenient (lower IoU)

    Returns:
        mean_prec (np.ndarray): shape (num_recall_points,) averaged over all classes
        recall_points (np.ndarray): equally spaced recall points (0 to 1)
    """
    if eval_type not in pr_details:
        raise ValueError(f"eval_type '{eval_type}' not found in pr_details. Available: {list(pr_details.keys())}")
    prec_arr = pr_details[eval_type]  # shape: (num_classes, num_diffs, num_overlaps, num_points)
    if prec_arr.ndim < 4:
        raise ValueError(f"Expected 4D array, got shape {prec_arr.shape}")

    num_diffs = prec_arr.shape[1]
    if diff_idx >= num_diffs:
        raise ValueError(f"diff_idx {diff_idx} out of range (max {num_diffs-1}). Available difficulty indices: 0..{num_diffs-1}")

    if overlap_idx >= prec_arr.shape[2]:
        raise ValueError(f"Overlap index {overlap_idx} out of range (max {prec_arr.shape[2]-1})")

    # Average over all classes (axis=0)
    mean_prec = np.mean(prec_arr[:, diff_idx, overlap_idx, :], axis=0)
    num_points = mean_prec.shape[0]
    recall_points = np.linspace(0, 1, num_points)
    return mean_prec, recall_points

def main():
    parser = argparse.ArgumentParser(description="从多个 pr_details.pkl 文件绘制平均 PR 曲线（mAP 曲线），支持中文标签")
    parser.add_argument('--files', nargs='+', required=True, help='pr_details.pkl 文件路径列表')
    parser.add_argument('--labels', nargs='+', required=True, help='每条曲线的标签（与文件顺序一致），支持中文')
    parser.add_argument('--eval_type', default='3d', choices=['3d', 'bev', 'bbox'], help='评估类型：3d / bev / bbox')
    parser.add_argument('--diff_idx', type=int, default=0, help='难度索引（0 = 第一个难度级别）')
    parser.add_argument('--overlap_idx', type=int, default=1, choices=[0,1],
                        help='重叠阈值索引：0 = 严格（高 IoU），1 = 宽松（默认，对应标准 mAP）')
    parser.add_argument('--output', default='map_curves.png', help='输出图片文件名（支持中文路径）')
    parser.add_argument('--dpi', type=int, default=300, help='图片分辨率 DPI')
    parser.add_argument('--title', default=None, help='自定义图表标题，支持中文')

    args = parser.parse_args()

    if len(args.files) != len(args.labels):
        raise ValueError("文件数量和标签数量必须一致。")

    plt.figure(figsize=(10, 8))
    for fpath, label in zip(args.files, args.labels):
        try:
            with open(fpath, 'rb') as f:
                pr_details = pickle.load(f)
            mean_prec, recall_points = compute_mean_precision(
                pr_details, args.eval_type, args.diff_idx, args.overlap_idx)
            plt.plot(recall_points, mean_prec, linewidth=2, label=label)
        except Exception as e:
            print(f"处理文件 {fpath} 时出错：{e}")
            continue

    if not plt.gca().get_legend_handles_labels()[1]:  # 没有成功添加曲线
        print("未绘制任何曲线，退出。")
        return

    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    if args.title:
        plt.title(args.title)
    else:
        plt.title(f'PR 曲线')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi)
    print(f"曲线图已保存至：{args.output}")

if __name__ == '__main__':
    main()