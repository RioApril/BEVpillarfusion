import os
import numpy as np
import torch
from mmdet3d.apis import inference_detector, init_model
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader
from tqdm import tqdm
import mmcv

# ========== 配置 ==========
config_file = '/home/vipuser/project/mmdetection3d/work_dirs/final_result/bevfusion/20260505_151518/vis_data/config.py'  # 你的配置文件路径
checkpoint_file = '/home/vipuser/project/mmdetection3d/work_dirs/final_result/bevfusion/epoch_10.pth'  # 训练好的权重文件
device = 'cpu'  # 或 'cuda:0'
root_dir = '/home/vipuser/project/mmdetection3d/data/view_of_delft_PUBLIC/radar_3frames'
output_dir = '/home/vipuser/project/mmdetection3d/kitti_predictions_bevfusion'          # 保存检测结果的文件夹
os.makedirs(output_dir, exist_ok=True)

# 初始化模型
model = init_model(config_file, checkpoint_file, device=device)

# 获取所有要推理的帧号（例如从 000000 到 01200）
frame_numbers = [f"{i:05d}" for i in range(10)]  # 根据需要调整范围

for frame_number in tqdm(frame_numbers, desc='生成 KITTI 检测文件'):
    # 构造点云路径（根据你的实际路径调整）
    point_cloud_path = os.path.join(root_dir, 'training', 'velodyne_reduced', f"{frame_number}.bin")
    if not os.path.exists(point_cloud_path):
        continue

    # 加载点云
    points = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, 7)
    use_dims = [0, 1, 2, 3, 5]
    points = points[:, use_dims]   # 变为 (N, 5)
    
    img_path = os.path.join(root_dir, 'training', 'image_2', f"{frame_number}.jpg")
    img = mmcv.imread(img_path)
    inputs = dict(points=points, img=[img])
    # 推理
    result = inference_detector(model, inputs)
    data_sample, _ = result
    pred = data_sample.pred_instances_3d

    bboxes_lidar = pred.bboxes_3d.tensor.cpu().numpy()
    scores = pred.scores_3d.cpu().numpy()
    labels = pred.labels_3d.cpu().numpy()

    # 过滤低分
    keep = scores > 0.3
    bboxes_lidar = bboxes_lidar[keep]
    labels = labels[keep]
    scores = scores[keep]

    if len(bboxes_lidar) == 0:
        # 无检测结果，创建空文件或跳过
        continue

    # 读取标定文件（每个帧对应一个标定文件）
    calib_file = os.path.join(root_dir, 'training', 'calib', f"{frame_number}.txt")
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
    calib = read_calib(calib_file)
    R0 = calib['R0_rect']
    Tr_velo_to_cam = calib['Tr_velo_to_cam']

    # 转换到相机坐标系
    def lidar_to_camera_boxes(boxes_lidar, R0, Tr_velo_to_cam):
        N = boxes_lidar.shape[0]
        # 中心点
        pts = boxes_lidar[:, :3]
        ones = np.ones((N, 1))
        pts_h = np.hstack([pts, ones])
        pts_cam = (R0 @ (Tr_velo_to_cam @ pts_h.T)).T   # 合并变换
        # 尺寸
        hwl = boxes_lidar[:, [5, 4, 3]]   # dz, dy, dx
        # 旋转
        yaw = boxes_lidar[:, 6]
        dir_lidar = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=1)
        dir_cam = (R0 @ (Tr_velo_to_cam[:, :3] @ dir_lidar.T)).T
        rot_y = np.arctan2(dir_cam[:, 0], dir_cam[:, 2])
        return np.column_stack([pts_cam, hwl, rot_y])

    boxes_cam = lidar_to_camera_boxes(bboxes_lidar, R0, Tr_velo_to_cam)

    # 获取图像尺寸（用于2D框投影）
    img_path = os.path.join(root_dir, 'testing', 'image_2', f"{frame_number}.jpg")
    if os.path.exists(img_path):
        import cv2
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
    else:
        # 如果没有图像，2D框填 -1
        img_w, img_h = -1, -1

    # 计算2D框：投影3D框的8个角点
    def compute_2d_bbox(box_cam, P2, img_w, img_h):
        # box_cam: [x, y, z, h, w, l, rot_y] 相机坐标系
        # 生成8个角点（相机坐标系）
        x, y, z, h, w, l, ry = box_cam
        R = np.array([[np.cos(ry), 0, np.sin(ry)],
                      [0, 1, 0],
                      [-np.sin(ry), 0, np.cos(ry)]])
        corners = np.array([
            [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2],
            [0, 0, 0, 0, -h, -h, -h, -h],
            [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        ])  # 3x8
        corners = R @ corners
        corners[0, :] += x
        corners[1, :] += y
        corners[2, :] += z
        # 投影到图像
        ones = np.ones((1, 8))
        corners_h = np.vstack([corners, ones])
        pts_img = P2 @ corners_h
        pts_img = pts_img[:2, :] / pts_img[2, :]
        # 计算最小外接矩形
        u = pts_img[0, :]
        v = pts_img[1, :]
        u_min, u_max = u.min(), u.max()
        v_min, v_max = v.min(), v.max()
        # 裁剪到图像边界
        if img_w > 0:
            u_min = max(0, u_min)
            u_max = min(img_w, u_max)
            v_min = max(0, v_min)
            v_max = min(img_h, v_max)
        return [u_min, v_min, u_max, v_max]

    # 读取投影矩阵 P2（从标定文件）
    P2 = calib['P2'].reshape(3, 4)

    # 生成 KITTI 格式行
    class_names = ['Pedestrian', 'Cyclist', 'Car']
    lines = []
    for i in range(len(boxes_cam)):
        box = boxes_cam[i]
        score = scores[i]
        cls_name = class_names[labels[i]]
        bbox_2d = compute_2d_bbox(box, P2, img_w, img_h)
        # 填写字段（truncated, occluded, alpha 可填 -1 或计算）
        # 简化处理：truncated=0, occluded=0, alpha=-10
        line = f"{cls_name} 0 0 -10 {bbox_2d[0]:.2f} {bbox_2d[1]:.2f} {bbox_2d[2]:.2f} {bbox_2d[3]:.2f} {box[4]:.2f} {box[5]:.2f} {box[3]:.2f} {box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[6]:.2f} {score:.4f}\n"
        lines.append(line)

    # 写入文件
    out_file = os.path.join(output_dir, f"{frame_number}.txt")
    with open(out_file, 'w') as f:
        f.writelines(lines)

print(f"KITTI 格式检测文件已保存到 {output_dir}")