import os
import numpy as np
import torch
import mmcv
from mmengine.config import Config
from mmdet3d.registry import DATASETS, MODELS, TRANSFORMS
from mmdet3d.apis import init_model
from tqdm import tqdm
import projects.BEVFusion.bevfusion.loading as _
from mmdet3d.registry import TRANSFORMS as MMDet3D_TRANSFORMS
import mmengine.registry

# 获取 mmdet3d 注册表中的类
cls = MMDet3D_TRANSFORMS.get('BEVLoadMultiViewImageFromFiles')
if cls is None:
    raise RuntimeError("Class not found in mmdet3d registry")
# 注册到 mmengine 的 TRANSFORMS
mmengine.registry.TRANSFORMS.register_module(name='BEVLoadMultiViewImageFromFiles', module=cls)

cls = MMDet3D_TRANSFORMS.get('LoadPointsFromFile')
if cls is None:
    raise RuntimeError("Class not found in mmdet3d registry")
# 注册到 mmengine 的 TRANSFORMS
mmengine.registry.TRANSFORMS.register_module(name='LoadPointsFromFile', module=cls)

cls = MMDet3D_TRANSFORMS.get('ImageAug3D')
if cls is None:
    raise RuntimeError("Class not found in mmdet3d registry")
# 注册到 mmengine 的 TRANSFORMS
mmengine.registry.TRANSFORMS.register_module(name='ImageAug3D', module=cls)

cls = MMDet3D_TRANSFORMS.get('PointsRangeFilter')
if cls is None:
    raise RuntimeError("Class not found in mmdet3d registry")
# 注册到 mmengine 的 TRANSFORMS
mmengine.registry.TRANSFORMS.register_module(name='PointsRangeFilter', module=cls)

cls = MMDet3D_TRANSFORMS.get('Pack3DDetInputs')
if cls is None:
    raise RuntimeError("Class not found in mmdet3d registry")
# 注册到 mmengine 的 TRANSFORMS
mmengine.registry.TRANSFORMS.register_module(name='Pack3DDetInputs', module=cls)

# 之后正常使用 mmengine 的 Compose
from mmengine.dataset import Compose

# print("BEVLoadMultiViewImageFromFiles" in TRANSFORMS.module_dict)
# ========== 配置 ==========
config_file = '/home/vipuser/project/mmdetection3d/work_dirs/final_result/bevfusion/20260505_151518/vis_data/config.py'  # 你的配置文件路径
checkpoint_file = '/home/vipuser/project/mmdetection3d/work_dirs/final_result/bevfusion/epoch_10.pth'  # 训练好的权重文件
device = 'cpu'   # 必须使用 GPU，cpu 会极慢且可能报错
root_dir = '/home/vipuser/project/mmdetection3d/data/view_of_delft_PUBLIC/radar_3frames'
output_dir = '/home/vipuser/project/mmdetection3d/kitti_predictions_bevfusion'          # 保存检测结果的文件夹
os.makedirs(output_dir, exist_ok=True)

# 加载配置
cfg = Config.fromfile(config_file)

# 构建测试 pipeline（从配置中复制 test_pipeline，并修正参数不兼容问题）
test_pipeline_cfg = cfg.test_pipeline.copy()

# 注意：这里手动处理可能的参数兼容问题
for i, trans_cfg in enumerate(test_pipeline_cfg):
    if trans_cfg['type'] == 'BEVLoadMultiViewImageFromFiles':
        # 删除不兼容的参数（如果存在）
        trans_cfg.pop('color_type', None)
        trans_cfg.pop('set_default_scale', None)
        # 确保必要的参数存在
        trans_cfg.setdefault('to_float32', True)
        trans_cfg.setdefault('num_views', 1)
        trans_cfg.setdefault('backend_args', None)
    elif trans_cfg['type'] == 'LoadPointsFromFile':
        # 确保有 coord_type
        trans_cfg.setdefault('coord_type', 'LIDAR')
        trans_cfg.setdefault('load_dim', 7)
        trans_cfg.setdefault('use_dim', [0,1,2,3,5])
    elif trans_cfg['type'] == 'ImageAug3D':
        # 可能存在 is_train 等参数，保留原样
        pass
    # 其他 transform 保持原样

# 创建 Compose 对象
test_pipeline = Compose(test_pipeline_cfg)

# 初始化模型
model = init_model(config_file, checkpoint_file, device=device)
model.eval()

# 要处理的帧号（示例 0 到 9）
frame_numbers = [f"{i:05d}" for i in range(10)]

# 辅助函数：读取标定文件
def read_calib(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                key, val = line.strip().split(':', 1)
                vals = [float(v) for v in val.strip().split()]
                if len(vals) == 12:
                    data[key] = np.array(vals).reshape(3, 4)
                elif len(vals) == 9:
                    data[key] = np.array(vals).reshape(3, 3)
                else:
                    data[key] = np.array(vals)
    return data

# 辅助函数：构建 pipeline 所需的 results 字典
def build_results(frame_number, root_dir):
    pc_path = os.path.join(root_dir, 'training', 'velodyne_reduced', f"{frame_number}.bin")
    img_path = os.path.join(root_dir, 'training', 'image_2', f"{frame_number}.jpg")
    calib_path = os.path.join(root_dir, 'training', 'calib', f"{frame_number}.txt")
    
    if not os.path.exists(pc_path) or not os.path.exists(img_path):
        return None
    
    calib = read_calib(calib_path)
    P2 = calib['P2'].reshape(3, 4)                       # 3x4
    Tr_velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)  # 3x4
    
    # 读取点云（保留所有列，后续 pipeline 会根据 use_dim 自动筛选）
    points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 7)
    
    # 构建 images 字典 (BEVLoadMultiViewImageFromFiles 需要)
    images = {
        'CAM2': {   # 默认相机名，与配置文件中的 default_cam_key 一致
            'img_path': img_path,
            'cam2img': P2.tolist(),
            'lidar2cam': Tr_velo_to_cam.tolist(),
        }
    }
    
    # 构建完整的 results
    results = {
        'lidar_points': {'lidar_path': pc_path, 'num_pts_feats': points.shape[1]},
        'images': images,
        'img_path': [img_path],          # 列表形式，长度为1
        'img_filename': [img_path],      # 某些 pipeline 可能需要
        'sample_idx': frame_number,
        'lidar_path': pc_path,
        'num_pts_feats': points.shape[1],
        # 标定信息（用于后续变换，非必须但建议）
        'cam2img': np.stack([np.eye(4, dtype=np.float32)]),  # 占位，实际会从 images 中获取
        'lidar2cam': np.stack([np.eye(4, dtype=np.float32)]),
        'lidar2img': np.stack([np.eye(4, dtype=np.float32)]),
        'ori_cam2img': np.stack([np.eye(4, dtype=np.float32)]),
    }
    return results, points

# 主循环
for frame_number in tqdm(frame_numbers, desc='生成 KITTI 检测文件'):
    results_and_points = build_results(frame_number, root_dir)
    if results_and_points is None:
        continue
    results, _ = results_and_points
    
    # 应用 pipeline，得到可直接送入模型的数据
    try:
        data = test_pipeline(results)
    except Exception as e:
        print(f"Pipeline 处理帧 {frame_number} 失败: {e}")
        continue
    
    # 模型预测
    with torch.no_grad():
        # data 通常包含 'inputs' 和 'data_samples'
        inputs = data['inputs']
        data_samples = data['data_samples']
        if not isinstance(data_samples, list):
            data_samples = [data_samples]
        # 推理
        preds = model.predict(inputs, data_samples)
    
    # 解析预测结果
    pred_sample = preds[0]
    if not hasattr(pred_sample, 'pred_instances_3d'):
        continue
    pred_inst = pred_sample.pred_instances_3d
    if pred_inst is None or len(pred_inst.bboxes_3d) == 0:
        continue
    
    bboxes_lidar = pred_inst.bboxes_3d.tensor.cpu().numpy()
    scores = pred_inst.scores_3d.cpu().numpy()
    labels = pred_inst.labels_3d.cpu().numpy()
    
    # 置信度过滤
    keep = scores > 0.3
    bboxes_lidar = bboxes_lidar[keep]
    labels = labels[keep]
    scores = scores[keep]
    
    if len(bboxes_lidar) == 0:
        continue
    
    # ========== 转换到相机坐标系并输出 KITTI 格式 ==========
    # 需要重新读取标定信息（之前 results 中可能有，但为了准确，重新获取）
    calib_file = os.path.join(root_dir, 'training', 'calib', f"{frame_number}.txt")
    calib = read_calib(calib_file)
    R0 = calib['R0_rect']
    Tr_velo_to_cam = calib['Tr_velo_to_cam']
    P2 = calib['P2'].reshape(3, 4)
    
    # lidar 框转相机坐标系（使用您原来的函数）
    def lidar_to_camera_boxes(boxes_lidar, R0, Tr_velo_to_cam):
        N = boxes_lidar.shape[0]
        pts = boxes_lidar[:, :3]
        ones = np.ones((N, 1))
        pts_h = np.hstack([pts, ones])
        pts_cam = (R0 @ (Tr_velo_to_cam @ pts_h.T)).T
        hwl = boxes_lidar[:, [5, 4, 3]]
        yaw = boxes_lidar[:, 6]
        dir_lidar = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=1)
        dir_cam = (R0 @ (Tr_velo_to_cam[:, :3] @ dir_lidar.T)).T
        rot_y = np.arctan2(dir_cam[:, 0], dir_cam[:, 2])
        return np.column_stack([pts_cam, hwl, rot_y])
    
    boxes_cam = lidar_to_camera_boxes(bboxes_lidar, R0, Tr_velo_to_cam)
    
    # 计算2D框（使用原来的 compute_2d_bbox 函数）
    def compute_2d_bbox(box_cam, P2, img_w, img_h):
        x, y, z, h, w, l, ry = box_cam
        R = np.array([[np.cos(ry), 0, np.sin(ry)],
                      [0, 1, 0],
                      [-np.sin(ry), 0, np.cos(ry)]])
        corners = np.array([
            [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2],
            [0, 0, 0, 0, -h, -h, -h, -h],
            [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        ])
        corners = R @ corners
        corners[0, :] += x
        corners[1, :] += y
        corners[2, :] += z
        ones = np.ones((1, 8))
        corners_h = np.vstack([corners, ones])
        pts_img = P2 @ corners_h
        pts_img = pts_img[:2, :] / pts_img[2, :]
        u_min, u_max = pts_img[0, :].min(), pts_img[0, :].max()
        v_min, v_max = pts_img[1, :].min(), pts_img[1, :].max()
        if img_w > 0:
            u_min = max(0, u_min)
            u_max = min(img_w, u_max)
            v_min = max(0, v_min)
            v_max = min(img_h, v_max)
        return [u_min, v_min, u_max, v_max]
    
    img_path = os.path.join(root_dir, 'training', 'image_2', f"{frame_number}.jpg")
    if os.path.exists(img_path):
        img = mmcv.imread(img_path)
        img_h, img_w = img.shape[:2]
    else:
        img_w, img_h = -1, -1
    
    class_names = ['Pedestrian', 'Cyclist', 'Car']
    lines = []
    for i in range(len(boxes_cam)):
        box = boxes_cam[i]
        score = scores[i]
        cls_name = class_names[labels[i]]
        bbox_2d = compute_2d_bbox(box, P2, img_w, img_h)
        line = f"{cls_name} 0 0 -10 {bbox_2d[0]:.2f} {bbox_2d[1]:.2f} {bbox_2d[2]:.2f} {bbox_2d[3]:.2f} {box[4]:.2f} {box[5]:.2f} {box[3]:.2f} {box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[6]:.2f} {score:.4f}\n"
        lines.append(line)
    
    out_file = os.path.join(output_dir, f"{frame_number}.txt")
    with open(out_file, 'w') as f:
        f.writelines(lines)

print("推理完成！")