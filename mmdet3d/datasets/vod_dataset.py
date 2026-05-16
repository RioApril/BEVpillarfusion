# Copyright (c) OpenMMLab. All rights reserved.
import copy
from os import path as osp
from typing import Callable, List, Union

import numpy as np
from mmengine import load

from mmdet3d.registry import DATASETS
from mmdet3d.structures import CameraInstance3DBoxes
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class VodDataset(Det3DDataset):
    # METAINFO 保持不变
    METAINFO = {
        'classes': ('Pedestrian', 'Cyclist', 'Car', 'bicycle', 'motor', 'truck', 'ride_other',
                    'bicycle_rack', 'rider', 'vehicle_other', 'ride_uncertain', 'moped_scooter', 'human_depiction'),
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True),
                 default_cam_key: str = 'CAM2',
                 load_type: str = 'frame_based',
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 pcd_limit_range: List[float] = [0, -25.6, -3, 51.2, 25.6, 2],
                 demo_load: bool = False,
                 remove_empty_gt_bboxes: bool = False,
                 bevfusion_compatible: bool = False,
                 **kwargs) -> None:
        self.pcd_limit_range = pcd_limit_range
        assert load_type in ('frame_based', 'mv_image_based', 'fov_image_based')
        self.load_type = load_type
        self.demo_load = demo_load
        self.remove_empty_gt_bboxes = remove_empty_gt_bboxes
        self.bevfusion_compatible = bevfusion_compatible
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            default_cam_key=default_cam_key,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert self.modality is not None
        assert box_type_3d.lower() in ('lidar', 'camera')

    # 原有的 _remove_empty_gt_bboxes 方法保持不变
    def _remove_empty_gt_bboxes(self, ann_info: dict) -> dict:
        filtered_annotations = {}
        filter_mask = ann_info['num_lidar_pts'] > 0
        for key in ann_info.keys():
            if key != 'instances':
                filtered_annotations[key] = (ann_info[key][filter_mask])
            else:
                filtered_annotations[key] = ann_info[key]
        return filtered_annotations

    def _build_camera_info(self, calib: dict, cam_key: str = 'CAM2') -> dict:
        """从 KITTI 风格的标定参数构建相机信息字典。"""
        P2 = np.array(calib['P2'], dtype=np.float32)
        R0_rect = np.array(calib['R0_rect'], dtype=np.float32)
        Tr_velo_to_cam = np.array(calib['Tr_velo_to_cam'], dtype=np.float32)

        # lidar2cam = R0_rect @ Tr_velo_to_cam (3x4) -> 补齐为4x4
        lidar2cam_34 = R0_rect @ Tr_velo_to_cam
        lidar2cam = np.eye(4, dtype=np.float32)
        lidar2cam[:3, :] = lidar2cam_34

        # cam2img = P2 (3x4) -> 补齐为4x4
        cam2img = np.eye(4, dtype=np.float32)
        cam2img[:3, :] = P2

        lidar2img = cam2img @ lidar2cam
        cam2lidar = np.linalg.inv(lidar2cam)

        return {
            'lidar2cam': np.ascontiguousarray(lidar2cam),
            'cam2img': np.ascontiguousarray(cam2img),
            'lidar2img': np.ascontiguousarray(lidar2img),
            'cam2lidar': np.ascontiguousarray(cam2lidar),
        }

    def parse_data_info(self, info: dict) -> dict:
        # 1. 处理 plane（如果需要，但 BEVFusion 模式下可以跳过）
        if self.modality['use_lidar'] and not self.bevfusion_compatible:
            # 原有的 plane 转换代码，但需要适配 info 结构（没有 images 字段）
            if 'plane' in info:
                # 从 calib 构建 lidar2cam
                if 'calib' in info:
                    P2 = np.array(info['calib']['P2'])
                    R0_rect = np.array(info['calib']['R0_rect'])
                    Tr_velo_to_cam = np.array(info['calib']['Tr_velo_to_cam'])
                    lidar2cam_34 = R0_rect @ Tr_velo_to_cam
                    lidar2cam = np.eye(4)
                    lidar2cam[:3, :] = lidar2cam_34
                else:
                    raise KeyError('Cannot find calib for plane conversion')
                reverse = np.linalg.inv(lidar2cam)
                plane = np.array(info['plane'])
                (plane_norm_cam, plane_off_cam) = (plane[:3], -plane[:3] * plane[3])
                plane_norm_lidar = (reverse[:3, :3] @ plane_norm_cam[:, None])[:, 0]
                plane_off_lidar = (reverse[:3, :3] @ plane_off_cam[:, None][:, 0] + reverse[:3, 3])
                plane_lidar = np.zeros(4, dtype=np.float32)
                plane_lidar[:3] = plane_norm_lidar
                plane_lidar[3] = -plane_norm_lidar.T @ plane_off_lidar
                info['plane'] = plane_lidar
            else:
                info['plane'] = None

        if self.load_type == 'fov_image_based' and self.load_eval_anns:
            # 这里原本期望 info['cam_instances']，但 KITTI 风格没有，可能需要适配
            pass

        # 2. 调用基类方法，获取基本 data_info（处理路径等）
        data_info = super().parse_data_info(info)

        # 3. BEVFusion 兼容模式：构建 images 字段和顶层矩阵
        if self.bevfusion_compatible and self.modality.get('use_camera', False):
            # ---- 新格式：从 info['images'] 读取 ----
            if 'images' in info and self.default_cam_key in info['images']:
                cam_info = info['images'][self.default_cam_key].copy()
                # 处理图像路径
                raw_path = cam_info.get('img_path', '')
                if not raw_path and 'image' in info and 'image_path' in info['image']:
                    raw_path = info['image']['image_path']
                if raw_path:
                    # 避免重复拼接
                    if osp.isabs(raw_path) or self.data_root in raw_path:
                        img_path = raw_path
                    else:
                        prefix = self.data_prefix.get(self.default_cam_key, 'training/image_2')
                        img_path = osp.join(self.data_root, prefix, raw_path)
                else:
                    img_path = ''
                cam_info['img_path'] = img_path

                # 确保必要矩阵存在（尤其 cam2lidar 和 lidar2img）
                if 'cam2lidar' not in cam_info and 'lidar2cam' in cam_info:
                    lidar2cam = np.array(cam_info['lidar2cam'])
                    if lidar2cam.shape == (3, 4):
                        lidar2cam_full = np.eye(4)
                        lidar2cam_full[:3, :] = lidar2cam
                        lidar2cam = lidar2cam_full
                    cam_info['cam2lidar'] = np.linalg.inv(lidar2cam).tolist()
                if 'lidar2img' not in cam_info and 'cam2img' in cam_info and 'lidar2cam' in cam_info:
                    cam2img = np.array(cam_info['cam2img'])
                    lidar2cam = np.array(cam_info['lidar2cam'])
                    if cam2img.shape == (3, 4):
                        cam2img_full = np.eye(4)
                        cam2img_full[:3, :] = cam2img
                        cam2img = cam2img_full
                    if lidar2cam.shape == (3, 4):
                        lidar2cam_full = np.eye(4)
                        lidar2cam_full[:3, :] = lidar2cam
                        lidar2cam = lidar2cam_full
                    lidar2img = cam2img @ lidar2cam
                    cam_info['lidar2img'] = lidar2img.tolist()

                data_info['images'] = {self.default_cam_key: cam_info}
                # 提升矩阵到顶层
                for key in ['lidar2img', 'cam2img', 'lidar2cam', 'cam2lidar']:
                    if key in cam_info:
                        arr = np.array(cam_info[key])
                        if arr.shape == (4, 4):
                            data_info[key] = arr[np.newaxis, ...]
                        elif arr.shape == (1, 4, 4):
                            data_info[key] = arr
                        elif arr.shape == (3, 4):
                            arr_full = np.eye(4)
                            arr_full[:3, :] = arr
                            data_info[key] = arr_full[np.newaxis, ...]
                        else:
                            raise ValueError(f'Unexpected shape for {key}: {arr.shape}')
            # ---- 旧格式：从 info['calib'] 构建 ----
            elif 'calib' in info:
                cam_info = self._build_camera_info(info['calib'], self.default_cam_key)
                raw_path = None
                if 'image' in info and 'image_path' in info['image']:
                    raw_path = info['image']['image_path']
                elif 'img_path' in info:
                    raw_path = info['img_path']
                if raw_path:
                    if osp.isabs(raw_path) or self.data_root in raw_path:
                        img_path = raw_path
                    else:
                        prefix = self.data_prefix.get(self.default_cam_key, 'training/image_2')
                        img_path = osp.join(self.data_root, prefix, raw_path)
                else:
                    img_path = ''
                cam_info['img_path'] = img_path
                data_info['images'] = {self.default_cam_key: cam_info}
                for key in ['lidar2img', 'cam2img', 'lidar2cam', 'cam2lidar']:
                    if key in cam_info:
                        arr = np.array(cam_info[key])
                        if arr.shape == (4, 4):
                            data_info[key] = arr[np.newaxis, ...]
                        elif arr.shape == (1, 4, 4):
                            data_info[key] = arr
                        else:
                            raise ValueError(f'Unexpected shape for {key}: {arr.shape}')
                    else:
                        raise KeyError(f'{key} missing in built camera info')
            else:
                raise KeyError('Missing camera calibration')

        return data_info

    # parse_ann_info 和 load_data_list 保持不变（原有逻辑）
    def parse_ann_info(self, info: dict) -> dict:
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)
            if self.load_type in ['fov_image_based', 'mv_image_based']:
                ann_info['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
                ann_info['gt_bboxes_labels'] = np.array(0, dtype=np.int64)
                ann_info['centers_2d'] = np.zeros((0, 2), dtype=np.float32)
                ann_info['depths'] = np.zeros((0), dtype=np.float32)

        ann_info = self._remove_dontcare(ann_info)
        if self.remove_empty_gt_bboxes:
            ann_info = self._remove_empty_gt_bboxes(ann_info)

        # 获取 lidar2cam 矩阵
        lidar2cam = None
        if 'images' in info and self.default_cam_key in info['images']:
            lidar2cam = np.array(info['images'][self.default_cam_key].get('lidar2cam'))
        if lidar2cam is None and 'calib' in info:
            # 旧格式兼容
            P2 = np.array(info['calib']['P2'])
            R0_rect = np.array(info['calib']['R0_rect'])
            Tr_velo_to_cam = np.array(info['calib']['Tr_velo_to_cam'])
            lidar2cam_34 = R0_rect @ Tr_velo_to_cam
            lidar2cam = np.eye(4)
            lidar2cam[:3, :] = lidar2cam_34
        if lidar2cam is None:
            raise KeyError('Cannot find lidar2cam for annotation conversion')

        gt_bboxes_3d = CameraInstance3DBoxes(
            ann_info['gt_bboxes_3d']).convert_to(self.box_mode_3d, np.linalg.inv(lidar2cam))
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info

    def load_data_list(self) -> List[dict]:
        annotations = load(self.ann_file)
        if self.demo_load:
            annotations['data_list'] = annotations['data_list'][:10]
        if not isinstance(annotations, dict):
            raise TypeError(f'...')
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('...')
        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']

        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        data_list = []
        for raw_data_info in raw_data_list:
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                data_list.append(data_info)
            elif isinstance(data_info, list):
                data_list.extend(data_info)
            else:
                raise TypeError('...')
        return data_list