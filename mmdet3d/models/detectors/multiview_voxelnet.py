# Copyright (c) OpenMMLab. All rights reserved.
import torch
from typing import Tuple

from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStage3DDetector

@MODELS.register_module()
class MultiViewVoxelNet(SingleStage3DDetector):
    """支持多视图体素输入的 VoxelNet，用于3D检测。

    与标准VoxelNet不同，此模型从数据预处理器接收两个视图的体素数据：
    - BEV视图（压缩z）：bev_voxels, bev_num_points, bev_coors
    - 侧视图（压缩y）：side_voxels, side_num_points, side_coors
    """

    def __init__(self,
                 voxel_encoder: ConfigType,
                 middle_encoder: ConfigType,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)

    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
        """从点云中提取特征，使用多视图体素编码器。

        Args:
            batch_inputs_dict (dict): 包含以下键：
                - bev_voxels: (N_bev, M, C) BEV支柱的点特征
                - bev_num_points: (N_bev,) BEV每个支柱的实际点数
                - bev_coors: (N_bev, 4) BEV支柱坐标，顺序 (batch, z, y, x)，z=0
                - side_voxels: (N_side, M, C) 侧视图支柱的点特征
                - side_num_points: (N_side,) 侧视图每个支柱的实际点数
                - side_coors: (N_side, 4) 侧视图支柱坐标，顺序 (batch, y, z, x)，y=0

        Returns:
            Tuple[Tensor]: 经过backbone和neck处理后的多尺度特征图。
        """
        # 获取两个视图的数据
        bev_voxels = batch_inputs_dict['bev_voxels']
        bev_num_points = batch_inputs_dict['bev_num_points']
        bev_coors = batch_inputs_dict['bev_coors']
        side_voxels = batch_inputs_dict['side_voxels']
        side_num_points = batch_inputs_dict['side_num_points']
        side_coors = batch_inputs_dict['side_coors']

        # 调用多视图编码器，得到融合后的BEV特征向量 (N_bev, C_out)
        voxel_features = self.voxel_encoder(
            bev_voxels, bev_num_points, bev_coors,
            side_voxels, side_num_points, side_coors)

        # 将BEV特征散列到伪图像
        batch_size = bev_coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, bev_coors, batch_size)

        # 通过backbone和neck
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x


@MODELS.register_module()
class ThreeViewVoxelNet(SingleStage3DDetector):
    def __init__(self,
                 voxel_encoder: ConfigType,
                 middle_encoder: ConfigType,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)

    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[torch.Tensor]:
        bev_voxels = batch_inputs_dict['bev_voxels']
        bev_num_points = batch_inputs_dict['bev_num_points']
        bev_coors = batch_inputs_dict['bev_coors']
        xoz_voxels = batch_inputs_dict['xoz_voxels']
        xoz_num_points = batch_inputs_dict['xoz_num_points']
        xoz_coors = batch_inputs_dict['xoz_coors']
        yoz_voxels = batch_inputs_dict['yoz_voxels']
        yoz_num_points = batch_inputs_dict['yoz_num_points']
        yoz_coors = batch_inputs_dict['yoz_coors']

        voxel_features = self.voxel_encoder(
            bev_voxels, bev_num_points, bev_coors,
            xoz_voxels, xoz_num_points, xoz_coors,
            yoz_voxels, yoz_num_points, yoz_coors)

        batch_size = bev_coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, bev_coors, batch_size)

        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x
