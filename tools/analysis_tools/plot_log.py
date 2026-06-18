# import re
# import argparse
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# def parse_log(log_path):
#     """解析mmengine日志，提取每个epoch的平均loss和验证mAP"""
#     epoch_losses = {}   # epoch -> list of losses
#     epoch_map = {}      # epoch -> mAP

#     with open(log_path, 'r') as f:
#         lines = f.readlines()

#     # 逐行解析训练loss
#     train_pattern = re.compile(r'Epoch\(train\)\s+\[(\d+)\].*?loss:\s+([0-9.]+)')
#     for line in lines:
#         m = train_pattern.search(line)
#         if m:
#             epoch = int(m.group(1))
#             loss = float(m.group(2))
#             epoch_losses.setdefault(epoch, []).append(loss)

#     # 解析验证mAP（Entire annotated area下的mAP）
#     map_pattern = re.compile(r'mAP:\s+([0-9.]+)')
#     entire_area_found = False
#     for i, line in enumerate(lines):
#         if 'Entire annotated area:' in line:
#             entire_area_found = True
#             continue
#         if entire_area_found and 'mAP:' in line:
#             m = map_pattern.search(line)
#             if m:
#                 # 向上查找最近一次 Epoch(val)
#                 epoch_val = None
#                 for j in range(i-1, max(0, i-50), -1):
#                     val_match = re.search(r'Epoch\(val\)\s+\[(\d+)\]', lines[j])
#                     if val_match:
#                         epoch_val = int(val_match.group(1))
#                         break
#                 if epoch_val is not None:
#                     epoch_map[epoch_val] = float(m.group(1))
#                 entire_area_found = False  # 重置，避免匹配到后面的strict mAP

#     # 计算每个epoch的平均loss
#     epochs = sorted(epoch_losses.keys())
#     avg_losses = [np.mean(epoch_losses[e]) for e in epochs]

#     # mAP数据（按epoch排序）
#     map_epochs = sorted(epoch_map.keys())
#     map_values = [epoch_map[e] for e in map_epochs]

#     return epochs, avg_losses, map_epochs, map_values

# def save_loss_curve(epochs, avg_losses, save_path):
#     """保存损失曲线图"""
#     plt.figure(figsize=(8, 5))
#     plt.plot(epochs, avg_losses, marker='o', linestyle='-', markersize=3, color='blue')
#     plt.xlabel('Epoch')
#     plt.ylabel('Average Training Loss')
#     plt.title('Training Loss vs. Epoch')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     plt.close()
#     print(f"损失曲线已保存至: {save_path}")

# def save_map_curve(map_epochs, map_values, save_path):
#     """保存mAP曲线图"""
#     plt.figure(figsize=(8, 5))
#     plt.plot(map_epochs, map_values, marker='s', linestyle='-', color='orange', markersize=4)
#     plt.xlabel('Epoch')
#     plt.ylabel('mAP (Entire Area)')
#     plt.title('Validation mAP vs. Epoch')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     plt.close()
#     print(f"mAP曲线已保存至: {save_path}")

# def main():
#     parser = argparse.ArgumentParser(description='从mmengine日志提取训练损失和验证mAP并绘制曲线')
#     parser.add_argument('log_file', type=str, help='mmengine日志文件路径')
#     parser.add_argument('--output_dir', type=str, default='.', help='输出图片的文件夹路径（默认为当前目录）')
#     args = parser.parse_args()

#     # 确保输出目录存在
#     os.makedirs(args.output_dir, exist_ok=True)

#     # 解析日志
#     epochs, avg_losses, map_epochs, map_values = parse_log(args.log_file)

#     # 打印汇总信息
#     print("训练loss汇总 (每个epoch平均):")
#     for e, loss in zip(epochs, avg_losses):
#         print(f"Epoch {e:2d}: {loss:.4f}")
#     print("\n验证mAP汇总:")
#     for e, m in zip(map_epochs, map_values):
#         print(f"Epoch {e:2d}: {m:.4f}")

#     # 保存曲线图
#     loss_img = os.path.join(args.output_dir, 'loss_curve.png')
#     map_img = os.path.join(args.output_dir, 'map_curve.png')
#     save_loss_curve(epochs, avg_losses, loss_img)
#     save_map_curve(map_epochs, map_values, map_img)

# if __name__ == '__main__':
#     main()
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.font_manager as fm

# ==================== 强制设置中文字体（直接指定文件路径） ====================
font_paths = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",   # 文泉驿微米黑
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",     # 文泉驿正黑
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # 思源黑体
]
chinese_font = None
for path in font_paths:
    if os.path.exists(path):
        chinese_font = fm.FontProperties(fname=path)
        # 添加到 matplotlib 字体管理器
        fm.fontManager.addfont(path)
        # 设为全局默认
        plt.rcParams['font.family'] = chinese_font.get_name()
        print(f"成功加载中文字体：{path}")
        break

if chinese_font is None:
    print("警告：未找到预设的中文字体文件，将尝试使用系统字体名称...")
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
else:
    plt.rcParams['axes.unicode_minus'] = False
# ============================================================================

def parse_log(log_path):
    """解析单个 mmengine 日志，返回 (epochs, avg_losses, map_epochs, map_values)"""
    epoch_losses = {}
    epoch_map = {}

    with open(log_path, 'r') as f:
        lines = f.readlines()

    train_pattern = re.compile(r'Epoch\(train\)\s+\[(\d+)\].*?loss:\s+([0-9.]+)')
    for line in lines:
        m = train_pattern.search(line)
        if m:
            epoch = int(m.group(1))
            loss = float(m.group(2))
            epoch_losses.setdefault(epoch, []).append(loss)

    map_pattern = re.compile(r'mAP:\s+([0-9.]+)')
    entire_area_found = False
    for i, line in enumerate(lines):
        if 'Entire annotated area:' in line:
            entire_area_found = True
            continue
        if entire_area_found and 'mAP:' in line:
            m = map_pattern.search(line)
            if m:
                epoch_val = None
                for j in range(i-1, max(0, i-50), -1):
                    val_match = re.search(r'Epoch\(val\)\s+\[(\d+)\]', lines[j])
                    if val_match:
                        epoch_val = int(val_match.group(1))
                        break
                if epoch_val is not None:
                    epoch_map[epoch_val] = float(m.group(1))
                entire_area_found = False

    epochs = sorted(epoch_losses.keys())
    avg_losses = [np.mean(epoch_losses[e]) for e in epochs]
    map_epochs = sorted(epoch_map.keys())
    map_values = [epoch_map[e] for e in map_epochs]

    return epochs, avg_losses, map_epochs, map_values

def draw_multiple_curves(data_list, x_label, y_label, title, save_path, y_lim=None):
    plt.figure(figsize=(10, 6))
    for data in data_list:
        if data['x'] and data['y']:
            plt.plot(data['x'], data['y'], marker='o', linestyle='-', markersize=3, label=data['label'])
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title)
    plt.grid(True)
    if y_lim:
        plt.ylim(y_lim)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', labelsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"曲线图已保存至: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='从多个 mmengine 日志提取训练损失和验证 mAP，并绘制对比曲线')
    parser.add_argument('log_files', type=str, nargs='+', help='一个或多个 mmengine 日志文件路径')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='每条曲线对应的标签（与日志文件顺序一致），若不提供则使用 DEFAULT_LABELS 硬编码列表')
    parser.add_argument('--output_dir', type=str, default='.', help='输出图片的文件夹路径（默认为当前目录）')
    parser.add_argument('--loss_ylim', type=float, nargs=2, default=None, help='损失曲线纵轴范围，如 --loss_ylim 0 2')
    parser.add_argument('--map_ylim', type=float, nargs=2, default=None, help='mAP 曲线纵轴范围，如 --map_ylim 0 1')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 硬编码默认标签（请根据实际修改）
    DEFAULT_LABELS = ["Baseline", "Method A", "Method B"]  # 可以改成中文如 ["基线", "方法1"]

    num_logs = len(args.log_files)

    if args.labels is not None:
        if len(args.labels) != num_logs:
            print(f"错误：--labels 数量 ({len(args.labels)}) 与日志文件数量 ({num_logs}) 不一致")
            return
        labels = args.labels
    else:
        if len(DEFAULT_LABELS) >= num_logs:
            labels = DEFAULT_LABELS[:num_logs]
        else:
            labels = DEFAULT_LABELS.copy()
            for i in range(len(DEFAULT_LABELS), num_logs):
                labels.append(f"Run {i+1}")

    loss_data_list = []
    map_data_list = []

    for log_path, label in zip(args.log_files, labels):
        print(f"正在解析: {log_path} (标签: {label})")
        epochs, avg_losses, map_epochs, map_values = parse_log(log_path)
        print(f"  训练 loss 共 {len(epochs)} 个 epoch: {epochs[0] if epochs else None} -> {epochs[-1] if epochs else None}")
        if avg_losses:
            print(f"    loss 范围: {min(avg_losses):.4f} ~ {max(avg_losses):.4f}")
        print(f"  验证 mAP 共 {len(map_epochs)} 个 epoch: {map_epochs[0] if map_epochs else None} -> {map_epochs[-1] if map_epochs else None}")
        if map_values:
            print(f"    mAP 范围: {min(map_values):.4f} ~ {max(map_values):.4f}")

        loss_data_list.append({'x': epochs, 'y': avg_losses, 'label': label})
        map_data_list.append({'x': map_epochs, 'y': map_values, 'label': label})

    if any(len(d['y']) > 0 for d in loss_data_list):
        loss_img = os.path.join(args.output_dir, 'loss_curves.png')
        draw_multiple_curves(loss_data_list, 'Epoch', '平均训练损失',
                             '训练损失随 Epoch 变化（多组对比）',
                             loss_img, y_lim=args.loss_ylim)
    else:
        print("警告：未找到任何有效的 loss 数据，跳过损失曲线绘制。")

    if any(len(d['y']) > 0 for d in map_data_list):
        map_img = os.path.join(args.output_dir, 'map_curves.png')
        draw_multiple_curves(map_data_list, 'Epoch', 'mAP (完整标注区域)',
                             '验证集 mAP 随 Epoch 变化（多组对比）',
                             map_img, y_lim=args.map_ylim)
    else:
        print("警告：未找到任何有效的 mAP 数据，跳过 mAP 曲线绘制。")

if __name__ == '__main__':
    main()