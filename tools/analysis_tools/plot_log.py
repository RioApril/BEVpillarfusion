import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_log(log_path):
    """解析mmengine日志，提取每个epoch的平均loss和验证mAP"""
    epoch_losses = {}   # epoch -> list of losses
    epoch_map = {}      # epoch -> mAP

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # 逐行解析训练loss
    train_pattern = re.compile(r'Epoch\(train\)\s+\[(\d+)\].*?loss:\s+([0-9.]+)')
    for line in lines:
        m = train_pattern.search(line)
        if m:
            epoch = int(m.group(1))
            loss = float(m.group(2))
            epoch_losses.setdefault(epoch, []).append(loss)

    # 解析验证mAP（Entire annotated area下的mAP）
    map_pattern = re.compile(r'mAP:\s+([0-9.]+)')
    entire_area_found = False
    for i, line in enumerate(lines):
        if 'Entire annotated area:' in line:
            entire_area_found = True
            continue
        if entire_area_found and 'mAP:' in line:
            m = map_pattern.search(line)
            if m:
                # 向上查找最近一次 Epoch(val)
                epoch_val = None
                for j in range(i-1, max(0, i-50), -1):
                    val_match = re.search(r'Epoch\(val\)\s+\[(\d+)\]', lines[j])
                    if val_match:
                        epoch_val = int(val_match.group(1))
                        break
                if epoch_val is not None:
                    epoch_map[epoch_val] = float(m.group(1))
                entire_area_found = False  # 重置，避免匹配到后面的strict mAP

    # 计算每个epoch的平均loss
    epochs = sorted(epoch_losses.keys())
    avg_losses = [np.mean(epoch_losses[e]) for e in epochs]

    # mAP数据（按epoch排序）
    map_epochs = sorted(epoch_map.keys())
    map_values = [epoch_map[e] for e in map_epochs]

    return epochs, avg_losses, map_epochs, map_values

def save_loss_curve(epochs, avg_losses, save_path):
    """保存损失曲线图"""
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, avg_losses, marker='o', linestyle='-', markersize=3, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Average Training Loss')
    plt.title('Training Loss vs. Epoch')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"损失曲线已保存至: {save_path}")

def save_map_curve(map_epochs, map_values, save_path):
    """保存mAP曲线图"""
    plt.figure(figsize=(8, 5))
    plt.plot(map_epochs, map_values, marker='s', linestyle='-', color='orange', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('mAP (Entire Area)')
    plt.title('Validation mAP vs. Epoch')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"mAP曲线已保存至: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='从mmengine日志提取训练损失和验证mAP并绘制曲线')
    parser.add_argument('log_file', type=str, help='mmengine日志文件路径')
    parser.add_argument('--output_dir', type=str, default='.', help='输出图片的文件夹路径（默认为当前目录）')
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 解析日志
    epochs, avg_losses, map_epochs, map_values = parse_log(args.log_file)

    # 打印汇总信息
    print("训练loss汇总 (每个epoch平均):")
    for e, loss in zip(epochs, avg_losses):
        print(f"Epoch {e:2d}: {loss:.4f}")
    print("\n验证mAP汇总:")
    for e, m in zip(map_epochs, map_values):
        print(f"Epoch {e:2d}: {m:.4f}")

    # 保存曲线图
    loss_img = os.path.join(args.output_dir, 'loss_curve.png')
    map_img = os.path.join(args.output_dir, 'map_curve.png')
    save_loss_curve(epochs, avg_losses, loss_img)
    save_map_curve(map_epochs, map_values, map_img)

if __name__ == '__main__':
    main()