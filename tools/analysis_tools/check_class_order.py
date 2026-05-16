# check_class_order.py - 安全版本
import pickle

# 加载训练集标注信息
with open('data/view_of_delft_PUBLIC/radar_3frames/kitti_infos_train.pkl', 'rb') as f:
    infos = pickle.load(f)

print("=== 基本信息 ===")
print(f"infos 类型：{type(infos)}")
print(f"infos 长度：{len(infos)}")

print("\n=== 第一条数据的键 ===")
info = infos[0]
print(f"info 类型：{type(info)}")
print(f"info 的键：{info.keys() if isinstance(info, dict) else '不是字典'}")

print("\n=== 第一条数据的完整结构（前 3 层）===")
def print_structure(obj, depth=0, max_depth=3):
    indent = "  " * depth
    if depth >= max_depth:
        print(f"{indent}...")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{indent}{k}: {type(v).__name__}")
            if isinstance(v, (dict, list)) and len(str(v)) < 200:
                print_structure(v, depth+1, max_depth)
    elif isinstance(obj, list):
        print(f"{indent}[list with {len(obj)} items]")
        if len(obj) > 0:
            print_structure(obj[0], depth+1, max_depth)
    else:
        print(f"{indent}{str(obj)[:100]}")

print_structure(info)

print("\n=== 尝试查找类别信息 ===")
# 尝试不同的可能路径
possible_keys = ['annos', 'annotation', 'labels', 'gt', 'name', 'class']
for key in possible_keys:
    if key in info:
        print(f"找到键 '{key}': {type(info[key])}")
        if isinstance(info[key], dict):
            print(f"  子键：{info[key].keys()}")
        elif isinstance(info[key], list) and len(info[key]) > 0:
            print(f"  前 3 项：{info[key][:3]}")