import torch
import os
import argparse
from collections import defaultdict

def analyze_layers_from_checkpoint(checkpoint_path, top_n=10, show_all=False):
    """
    分析 .pth 文件的层结构和参数量
    
    Args:
        checkpoint_path: .pth 文件路径
        top_n: 每个模块显示的最大层数
        show_all: 是否显示所有层（默认只显示前 top_n 层）
    """
    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误：文件 {checkpoint_path} 不存在")
        return None

    print(f"🔍 正在分析：{checkpoint_path} ...\n")
    
    # 加载 checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"❌ 加载失败：{e}")
        return None

    # 提取 state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("✓ 检测到 'state_dict' 键")
    elif isinstance(checkpoint, dict):
        state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
        print("✓ 直接使用顶层字典中的 Tensor")
    else:
        print("❌ 无法识别的 .pth 文件格式")
        return None

    # 按模块前缀分组
    modules = defaultdict(list)
    for name, tensor in state_dict.items():
        # 去除 'module.' 前缀 (DDP 训练常见)
        if name.startswith('module.'):
            name = name[7:]
        # 去除 'ema_state_dict.' 前缀 (EMA 模型)
        if name.startswith('ema_state_dict.'):
            name = name[15:]
            
        parts = name.split('.')
        # 取前 2 级作为模块名 (如 'backbone', 'neck', 'bbox_head')
        module_name = '.'.join(parts[:2]) if len(parts) > 2 else parts[0]
        modules[module_name].append({
            'full_name': name,
            'shape': tuple(tensor.shape),
            'params': tensor.numel(),
            'dtype': str(tensor.dtype)
        })

    # 输出统计
    print("\n" + "=" * 90)
    print(f"📦 模型权重文件：{os.path.basename(checkpoint_path)}")
    print(f"📊 总层数：{len(state_dict)}")
    print(f"💾 文件大小：{os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")
    print("=" * 90)

    total_params = 0
    module_summary = []
    
    for module_name, layers in sorted(modules.items()):
        module_params = sum(l['params'] for l in layers)
        total_params += module_params
        module_summary.append((module_name, len(layers), module_params))
        
        print(f"\n【{module_name}】")
        print(f"  ├─ 子层数：{len(layers)}")
        print(f"  ├─ 参数量：{module_params / 1e6:.4f} M")
        print(f"  └─ 权重详情:")
        
        display_count = len(layers) if show_all else min(top_n, len(layers))
        for i, layer in enumerate(layers[:display_count]):
            prefix = "    ├─" if i < display_count - 1 else "    └─"
            print(f"  {prefix} {layer['full_name']:<55} {str(layer['shape']):<25} {layer['params']:>12,}")
        
        if not show_all and len(layers) > top_n:
            print(f"    ... 还有 {len(layers) - top_n} 层未显示 (使用 --show-all 查看全部)")

    # 输出汇总表格
    print("\n" + "=" * 90)
    print("📋 模块汇总")
    print("=" * 90)
    print(f"{'模块名':<30} {'层数':>8} {'参数量 (M)':>12} {'占比':>10}")
    print("-" * 90)
    
    for module_name, layer_count, params in sorted(module_summary, key=lambda x: x[2], reverse=True):
        percentage = (params / total_params * 100) if total_params > 0 else 0
        print(f"{module_name:<30} {layer_count:>8} {params / 1e6:>12.4f} {percentage:>9.2f}%")
    
    print("-" * 90)
    print(f"{'TOTAL':<30} {len(state_dict):>8} {total_params / 1e6:>12.4f} {'100.00%':>10}")
    print("=" * 90)
    
    # 参数量单位转换
    if total_params > 1e9:
        print(f"\n💡 总参数量：{total_params / 1e9:.2f} B ({total_params:,})")
    elif total_params > 1e6:
        print(f"\n💡 总参数量：{total_params / 1e6:.2f} M ({total_params:,})")
    else:
        print(f"\n💡 总参数量：{total_params:,}")
    
    print("=" * 90 + "\n")
    
    return modules

def main():
    parser = argparse.ArgumentParser(
        description='分析 PyTorch 模型 .pth 文件的层结构和参数量',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python analyze_checkpoint.py your_model.pth
  python analyze_checkpoint.py your_model.pth --top-n 20
  python analyze_checkpoint.py your_model.pth --show-all
  python analyze_checkpoint.py your_model.pth -o result.txt
        """
    )
    
    parser.add_argument(
        'checkpoint',
        type=str,
        help='模型 .pth 文件路径'
    )
    
    parser.add_argument(
        '--top-n', '-n',
        type=int,
        default=10,
        help='每个模块显示的最大层数 (默认：10)'
    )
    
    parser.add_argument(
        '--show-all', '-a',
        action='store_true',
        help='显示所有层，不限制数量'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='将结果保存到文件 (默认：仅输出到终端)'
    )
    
    args = parser.parse_args()
    
    # 执行分析
    if args.output:
        # 重定向输出到文件
        import sys
        original_stdout = sys.stdout
        with open(args.output, 'w', encoding='utf-8') as f:
            sys.stdout = f
            analyze_layers_from_checkpoint(args.checkpoint, args.top_n, args.show_all)
            sys.stdout = original_stdout
        print(f"✓ 结果已保存到：{args.output}")
    else:
        analyze_layers_from_checkpoint(args.checkpoint, args.top_n, args.show_all)

if __name__ == '__main__':
    main()