# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
from mmcv.cnn import build_norm_layer
from mmcv.ops import DynamicScatter
from torch import Tensor, nn

from mmdet3d.registry import MODELS
from .utils import PFNLayer, get_paddings_indicator

@MODELS.register_module()
class ViewPillarFeatureNet(nn.Module):
    """
    支持三种视图：
    - 'bev'    : 压缩z，保留x,y。coors顺序 (batch, z, y, x)，z=0
    - 'side_xz': 压缩y，保留x,z。coors顺序 (batch, y, z, x)，y=0
    - 'side_yz': 压缩x，保留y,z。coors顺序 (batch, x, z, y)，x=0
    """
    def __init__(self,
                 view_type: str = 'bev',          # 'bev', 'side_xz', 'side_yz'
                 in_channels: int = 4,
                 feat_channels: tuple = (64,),
                 with_distance: bool = False,
                 with_cluster_center: bool = True,
                 with_voxel_center: bool = True,
                 voxel_size: Tuple[float] = (0.2, 0.2, 4),
                 point_cloud_range: Tuple[float] = (0, -40, -3, 70.4, 40, 1),
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode: str = 'max',
                 legacy: bool = True):
        super().__init__()
        self.view_type = view_type
        self.legacy = legacy
        self.vx, self.vy, self.vz = voxel_size
        self.point_cloud_range = point_cloud_range

        # 根据视图类型设置坐标索引（用于voxel center）
        if view_type == 'bev':
            # 保留 x, y；压缩 z
            self.x_idx = 3
            self.y_idx = 2
            self.z_idx = 1   # 被压缩，z坐标索引但实际不用
            self.compressed_dims = ['z']
        elif view_type == 'side_xz':
            # 保留 x, z；压缩 y
            self.x_idx = 3
            self.y_idx = 1   # 被压缩，y坐标索引恒为0
            self.z_idx = 2
            self.compressed_dims = ['y']
        elif view_type == 'side_yz':
            # 保留 y, z；压缩 x
            self.x_idx = 1   # 被压缩，x坐标索引恒为0
            self.y_idx = 3
            self.z_idx = 2
            self.compressed_dims = ['x']
        else:
            raise ValueError(f"Unsupported view_type: {view_type}")

        # 输入特征维度装饰
        base_in_channels = in_channels
        if with_cluster_center:
            base_in_channels += 3
        if with_voxel_center:
            base_in_channels += 3
        if with_distance:
            base_in_channels += 1
        self.in_channels = base_in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center

        # 构建PFN层
        feat_channels = [self.in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            last_layer = (i == len(feat_channels) - 2)
            pfn_layers.append(PFNLayer(in_filters, out_filters, norm_cfg=norm_cfg,
                                       last_layer=last_layer, mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # 预先计算偏移量
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]

    def forward(self, features: torch.Tensor, num_points: torch.Tensor,
                coors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (N, M, C)  每个pillar内最多M个点的特征
            num_points: (N,)      每个pillar内的实际点数
            coors: (N, 4)         每个pillar的坐标索引，顺序由view_type决定
        Returns:
            pillar_features: (N, C_out)
        """
        features_ls = [features]
        dtype = features.dtype

        # 1. 聚类中心偏移
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # 2. 体素中心偏移
        if self._with_voxel_center:
            # 根据视图类型计算格子中心坐标
            if self.view_type == 'bev' or self.view_type == 'side_xz' or self.view_type == 'side_yz':
                # 计算 x, y 中心，z 中心不使用
                center_x = coors[:, self.x_idx].to(dtype).unsqueeze(1) * self.vx + self.x_offset
                center_y = coors[:, self.y_idx].to(dtype).unsqueeze(1) * self.vy + self.y_offset
                center_z = coors[:, self.z_idx].to(dtype).unsqueeze(1) * self.vz + self.z_offset
                f_center = features[:, :, :3] - torch.stack([center_x, center_y, center_z], dim=-1)
            # elif self.view_type == 'side_xz':
            #     # 保留 x 和 z 的中心偏移，y 方向不偏移
            #     center_x = coors[:, self.x_idx].to(dtype).unsqueeze(1) * self.vx + self.x_offset
            #     center_z = coors[:, self.z_idx].to(dtype).unsqueeze(1) * self.vz + self.z_offset
            #     f_center = torch.zeros_like(features[:, :, :3])
            #     f_center[:, :, 0] = features[:, :, 0] - center_x   # x方向偏移
            #     f_center[:, :, 1] = features[:, :, 1]               # y保持不变
            #     f_center[:, :, 2] = features[:, :, 2] - center_z   # z方向偏移
            # elif self.view_type == 'side_yz':
            #     # 保留 y 和 z 的中心偏移，x 方向不偏移
            #     center_y = coors[:, self.y_idx].to(dtype).unsqueeze(1) * self.vy + self.y_offset
            #     center_z = coors[:, self.z_idx].to(dtype).unsqueeze(1) * self.vz + self.z_offset
            #     f_center = torch.zeros_like(features[:, :, :3])
            #     f_center[:, :, 0] = features[:, :, 0]               # x保持不变
            #     f_center[:, :, 1] = features[:, :, 1] - center_y   # y方向偏移
            #     f_center[:, :, 2] = features[:, :, 2] - center_z   # z方向偏移
            else:
                raise ValueError(f"Unsupported view_type: {self.view_type}")
            features_ls.append(f_center)

        # 3. 距离特征
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # 合并特征
        features = torch.cat(features_ls, dim=-1)

        # 掩盖空pillar中的点
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # 通过PFN层
        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze(1)

@MODELS.register_module()
class SpatialChannelFusion(nn.Module):
    """
    融合 BEV 特征 (Y, X, C_bev) 和侧视图 (Z, X, C_side)
    以 X 为 batch，Y 为查询序列，Z 为键/值序列。
    输出 (Y, X, out_ch)
    """
    def __init__(self, in_ch_bev: int, in_ch_side: int, out_ch: int,
                 d_model: int = 128, n_heads: int = 4, dropout: float = 0.1,
                 max_len: int = 200):
        super().__init__()
        self.d_model = d_model
        self.out_ch = out_ch

        self.proj_bev = nn.Linear(in_ch_bev, d_model)
        self.proj_side = nn.Linear(in_ch_side, d_model)

        # 可学习位置编码
        self.pos_enc_y = nn.Parameter(torch.randn(max_len, 1, d_model) * 0.02)  # 用于查询 (Y序列)
        self.pos_enc_z = nn.Parameter(torch.randn(max_len, 1, d_model) * 0.02)  # 用于键/值 (Z序列)

        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.out_proj = nn.Linear(d_model, out_ch)

    def forward(self, bev_feat: torch.Tensor, side_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_feat:  (Y, X, C_bev)   # Y: H (高度格子数), X: W (宽度格子数)
            side_feat: (Z, X, C_side)   # Z: L (深度格子数), X: W
        Returns:
            fused:     (Y, X, out_ch)
        """
        Y, X, _ = bev_feat.shape
        Z, X2, _ = side_feat.shape
        assert X == X2, f"X维度必须相同，得到 {X} 和 {X2}"

        # 投影到 d_model
        q = self.proj_bev(bev_feat)   # (Y, X, D)
        k = self.proj_side(side_feat) # (Z, X, D)
        v = k  # 值与键共享投影，也可单独定义

        # 添加位置编码
        # pos_enc_y: (max_len, 1, D) -> 取前 Y 个，广播到 (Y, X, D)
        q = q + self.pos_enc_y[:Y]   # (Y, 1, D) 广播到 (Y, X, D)
        k = k + self.pos_enc_z[:Z]   # (Z, 1, D) 广播到 (Z, X, D)
        v = v + self.pos_enc_z[:Z]

        # 多头注意力
        # 输入形状已满足 (seq_len, batch, dim): q (Y, X, D), k (Z, X, D), v (Z, X, D)
        attn_out, _ = self.attention(q, k, v)   # 输出 (Y, X, D)
        attn_out = self.norm1(attn_out + q)

        # FFN
        ffn_out = self.ffn(attn_out)            # (Y, X, D)
        out = self.norm2(ffn_out + attn_out)    # (Y, X, D)

        # 输出投影
        out = self.out_proj(out)                 # (Y, X, out_ch)
        return out

@MODELS.register_module()
class MultiViewPillarFeatureNet(nn.Module):
    def __init__(self,
                 # BEV视图参数
                 bev_in_channels: int = 4,
                 bev_feat_channels: tuple = (64,),
                 # 侧视图参数 (xoz平面)
                 side_in_channels: int = 4,
                 side_feat_channels: tuple = (32,),
                 # 公共参数
                 with_distance: bool = False,
                 with_cluster_center: bool = True,
                 with_voxel_center: bool = True,
                 bev_voxel_size: Tuple[float] = (0.2, 0.2, 4),
                 side_voxel_size: Tuple[float] = (0.2, 4, 0.2),
                 point_cloud_range: Tuple[float] = (0, -40, -3, 70.4, 40, 1),
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode: str = 'max',
                 legacy: bool = True,
                 # 融合模块参数
                 d_model: int = 128,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 # 新增卷积参数
                 with_convolution: bool = False,
                 conv_kernel_size: int = 3):
        super().__init__()
        self.with_convolution = with_convolution
        self.conv_kernel_size = conv_kernel_size

        self.point_cloud_range = point_cloud_range
        x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range

        # BEV网格尺寸 (Y, X)
        self.H = int((y_max - y_min) / bev_voxel_size[1])
        self.W = int((x_max - x_min) / bev_voxel_size[0])

        # 侧视图网格尺寸 (Z, X)
        self.L = int((z_max - z_min) / side_voxel_size[2])
        side_W = int((x_max - x_min) / side_voxel_size[0])
        assert self.W == side_W, \
            f"X维度格子数不一致: BEV vx={bev_voxel_size[0]} -> W={self.W}, " \
            f"side vx={side_voxel_size[0]} -> side_W={side_W}"

        # 创建两个视图的PFN
        self.bev_pfn = ViewPillarFeatureNet(
            view_type='bev',
            in_channels=bev_in_channels,
            feat_channels=bev_feat_channels,
            with_distance=with_distance,
            with_cluster_center=with_cluster_center,
            with_voxel_center=with_voxel_center,
            voxel_size=bev_voxel_size,
            point_cloud_range=point_cloud_range,
            norm_cfg=norm_cfg,
            mode=mode,
            legacy=legacy
        )
        self.side_pfn = ViewPillarFeatureNet(
            view_type='side_xz',
            in_channels=side_in_channels,
            feat_channels=side_feat_channels,
            with_distance=with_distance,
            with_cluster_center=with_cluster_center,
            with_voxel_center=with_voxel_center,
            voxel_size=side_voxel_size,
            point_cloud_range=point_cloud_range,
            norm_cfg=norm_cfg,
            mode=mode,
            legacy=legacy
        )

        # 可选卷积层（扩大感受野）
        if self.with_convolution:
            # BEV卷积层 (H, W)
            bev_in_ch = bev_feat_channels[-1]
            self.bev_conv = nn.Sequential(
                nn.Conv2d(bev_in_ch, bev_in_ch, kernel_size=conv_kernel_size,
                          stride=1, padding=conv_kernel_size // 2, bias=False),
                nn.BatchNorm2d(bev_in_ch),
                nn.ReLU(inplace=True)
            )
            # 侧视图卷积层 (L, W)
            side_in_ch = side_feat_channels[-1]
            self.side_conv = nn.Sequential(
                nn.Conv2d(side_in_ch, side_in_ch, kernel_size=conv_kernel_size,
                          stride=1, padding=conv_kernel_size // 2, bias=False),
                nn.BatchNorm2d(side_in_ch),
                nn.ReLU(inplace=True)
            )

        # 融合模块
        max_len = max(self.H, self.L)
        self.fusion = SpatialChannelFusion(
            in_ch_bev=bev_feat_channels[-1],
            in_ch_side=side_feat_channels[-1],
            out_ch=bev_feat_channels[-1],
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_len=max_len
        )
        
        self.post_fusion_norm = nn.LayerNorm(bev_feat_channels[-1]) # ADD

    def forward(self,
                bev_features, bev_num_points, bev_coors,
                side_features, side_num_points, side_coors):
        """
        Args:
            bev_features:  (N_bev, M, C)
            bev_num_points: (N_bev,)
            bev_coors:      (N_bev, 4)    顺序 (batch, z, y, x)  z=0
            side_features:  (N_side, M, C)
            side_num_points: (N_side,)
            side_coors:     (N_side, 4)    顺序 (batch, y, z, x)  y=0
        Returns:
            fused_bev_map: (batch, H, W, C_out)
        """
        device = bev_features.device

        # 1. 通过PFN得到pillar特征
        bev_pillar_feat = self.bev_pfn(bev_features, bev_num_points, bev_coors)
        side_pillar_feat = self.side_pfn(side_features, side_num_points, side_coors)

        # 2. 计算真实的batch大小
        batch_size = int(bev_coors[:, 0].max().item()) + 1

        # 3. 散列到特征图
        # BEV散列: (batch, H, W, C_bev)
        bev_map = torch.zeros(batch_size, self.H, self.W, bev_pillar_feat.shape[-1], device=device)
        bev_y = bev_coors[:, 2].long()
        bev_x = bev_coors[:, 3].long()
        batch_idx = bev_coors[:, 0].long()
        bev_map[batch_idx, bev_y, bev_x, :] = bev_pillar_feat

        # 侧视图散列: (batch, L, W, C_side)
        side_map = torch.zeros(batch_size, self.L, self.W, side_pillar_feat.shape[-1], device=device)
        side_z = side_coors[:, 2].long()
        side_x = side_coors[:, 3].long()
        batch_idx_side = side_coors[:, 0].long()
        side_map[batch_idx_side, side_z, side_x, :] = side_pillar_feat

        # 4. 可选卷积：扩大感受野
        if self.with_convolution:
            # BEV: (batch, H, W, C) -> (batch, C, H, W)
            bev_map = bev_map.permute(0, 3, 1, 2).contiguous()
            bev_map = self.bev_conv(bev_map)
            bev_map = bev_map.permute(0, 2, 3, 1).contiguous()

            # Side: (batch, L, W, C) -> (batch, C, L, W)
            side_map = side_map.permute(0, 3, 1, 2).contiguous()
            side_map = self.side_conv(side_map)
            side_map = side_map.permute(0, 2, 3, 1).contiguous()

        # 5. 融合（逐batch）
        fused_list = []
        for b in range(batch_size):
            fused = self.fusion(bev_map[b], side_map[b])   # (H, W, C_out)
            fused = self.post_fusion_norm(fused) # ADD
            fused_list.append(fused.unsqueeze(0))
        fused = torch.cat(fused_list, dim=0)  # (batch, H, W, C_out)

        return fused

@MODELS.register_module()
class PillarFeatureNet(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels: Optional[int] = 4,
                 feat_channels: Optional[tuple] = (64, ),
                 with_distance: Optional[bool] = False,
                 with_cluster_center: Optional[bool] = True,
                 with_voxel_center: Optional[bool] = True,
                 voxel_size: Optional[Tuple[float]] = (0.2, 0.2, 4),
                 point_cloud_range: Optional[Tuple[float]] = (0, -40, -3, 70.4,
                                                              40, 1),
                 norm_cfg: Optional[dict] = dict(
                     type='BN1d', eps=1e-3, momentum=0.01),
                 with_height: Optional[bool] = False,
                 with_intensity: Optional[bool] = False,
                 with_velocity: Optional[bool] = False,
                 mode: Optional[str] = 'max',
                 legacy: Optional[bool] = True):
        super(PillarFeatureNet, self).__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self._with_height = with_height
        self._with_intensity = with_intensity
        self._with_velocity = with_velocity
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

        self.z_min_range = point_cloud_range[2]
        self.z_max_range = point_cloud_range[5]
        self.z_total_range = self.z_max_range - self.z_min_range

    def forward(self, features: Tensor, num_points: Tensor, coors: Tensor,
                *args, **kwargs) -> Tensor:
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :3])
                f_center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = features[:, :, 2] - (
                    coors[:, 1].to(dtype).unsqueeze(1) * self.vz +
                    self.z_offset)
            else:
                f_center = features[:, :, :3]
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = f_center[:, :, 2] - (
                    coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                    self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)
        
        if self._with_height:
            original_z = features[:,:,2:3]
            z_for_stats = original_z.clone()
            mask_bool = (mask.squeeze(-1) > 0)
            z_min_input = torch.where(mask_bool.unsqueeze(-1), z_for_stats, torch.full_like(z_for_stats, 1e9))
            z_max_input = torch.where(mask_bool.unsqueeze(-1), z_for_stats, torch.full_like(z_for_stats, -1e9))
            z_min = torch.min(z_min_input, dim=1)[0]
            z_max = torch.max(z_max_input, dim=1)[0]

            z_sum = torch.sum(z_for_stats * mask, dim=1)
            valid_counts = num_points.clamp(min=1).unsqueeze(-1).type_as(features)
            z_mean = z_sum / valid_counts
            z_range = z_max - z_min
            
            z_min_norm = (z_min - self.z_min_range) / (self.z_total_range + 1e-6)
            z_max_norm = (z_max - self.z_min_range) / (self.z_total_range + 1e-6)
            z_mean_norm = (z_mean - self.z_min_range) / (self.z_total_range + 1e-6)
            z_range_norm = z_range / (self.z_total_range + 1e-6)

            z_min_norm = torch.clamp(z_min_norm, 0, 1)
            z_max_norm = torch.clamp(z_max_norm, 0, 1)
            z_mean_norm = torch.clamp(z_mean_norm, 0, 1)
            z_range_norm = torch.clamp(z_range_norm, 0, 1)

            stats_feat = torch.cat([z_min_norm, z_max_norm, z_mean_norm, z_range_norm], dim=1)
            # 根据 features 的维度进行拼接
            if features.dim() == 3:
                # 假设 features 是 (N, 1, C) 或 (N, M, C)，我们将 stats_feat 扩展到相同维度
                stats_feat = stats_feat.unsqueeze(1)  # (N, 1, 4)
                # 如果 M > 1，可以 expand 到 (N, M, 4)（但通常 M=1，所以 unsqueeze 足够）
                if features.size(1) != 1:
                    stats_feat = stats_feat.expand(-1, features.size(1), -1)
                features = torch.cat([features, stats_feat], dim=2)  # 在通道维度拼接
            else:  # features.dim() == 2
                features = torch.cat([features, stats_feat], dim=1)  # (N, C+4)

        if self._with_intensity:
            raw_r = features[:, :, 3:4].clone()
            mask_2d = get_paddings_indicator(num_points, raw_r.shape[1], axis=0) # (N, M)
            valid_counts = num_points.clamp(min=1).unsqueeze(-1).type_as(raw_r)

            # Mean
            r_sum = torch.sum(raw_r * mask_2d.unsqueeze(-1), dim=1)
            r_mean = r_sum / valid_counts
            
            # Variance (关键！区分材质均匀度)
            r_sq_sum = torch.sum((raw_r ** 2) * mask_2d.unsqueeze(-1), dim=1)
            r_var = r_sq_sum / valid_counts - (r_mean ** 2)
            r_var = torch.clamp(r_var, min=0) # 防止负数

            r_mean_norm = r_mean / 255.0
            r_var_norm = r_var / 255.0 # 简单缩放
            r_stats = torch.cat([r_mean_norm, r_var_norm], dim=1) # (N, 4)

            # 根据 features 的维度进行拼接
            if features.dim() == 3:
                # 假设 features 是 (N, 1, C) 或 (N, M, C)，我们将 stats_feat 扩展到相同维度
                r_stats_feat = r_stats.unsqueeze(1)  # (N, 1, 4)
                # 如果 M > 1，可以 expand 到 (N, M, 4)（但通常 M=1，所以 unsqueeze 足够）
                if features.size(1) != 1:
                    r_stats_feat = r_stats_feat.expand(-1, features.size(1), -1)
                features = torch.cat([features, r_stats_feat], dim=2)  # 在通道维度拼接
            else:  # features.dim() == 2
                features = torch.cat([features, r_stats_feat], dim=1)  # (N, C+4)

        if self._with_velocity:
            raw_v = features[:, :, 5:6].clone() # Shape: (N, M, 1)
            mask_2d = get_paddings_indicator(num_points, raw_v.shape[1], axis=0) # (N, M)
            valid_counts = num_points.clamp(min=1).unsqueeze(-1).type_as(raw_v)

            # Mean Velocity
            v_sum = torch.sum(raw_v * mask_2d.unsqueeze(-1), dim=1)
            v_mean = v_sum / valid_counts # (N, 1)
            
            # Velocity Variance (方差 = E[x^2] - (E[x])^2)
            v_sq_sum = torch.sum((raw_v ** 2) * mask_2d.unsqueeze(-1), dim=1)
            v_var = v_sq_sum / valid_counts - (v_mean ** 2)
            v_var = torch.clamp(v_var, min=0) # 防止数值误差导致负数
            
            # 归一化 (假设速度范围 -10 ~ 10 m/s)
            v_mean_norm = (v_mean + 10.0) / 20.0 
            v_var_norm = v_var / 20.0 # 简单归一化
            
            v_stats = torch.cat([v_mean_norm, v_var_norm], dim=1) # (N, 2)

            if features.dim() == 3:
                # 假设 features 是 (N, 1, C) 或 (N, M, C)，我们将 stats_feat 扩展到相同维度
                v_stats_feat = v_stats.unsqueeze(1)  # (N, 1, 2)
                # 如果 M > 1，可以 expand 到 (N, M, 2)（但通常 M=1，所以 unsqueeze 足够）
                if features.size(1) != 1:
                    v_stats_feat = v_stats_feat.expand(-1, features.size(1), -1)
                features = torch.cat([features, v_stats_feat], dim=2)  # 在通道维度拼接
            else:  # features.dim() == 2
                features = torch.cat([features, v_stats_feat], dim=1)  # (N, C+2)

        return features.squeeze(1)

import torch.nn as nn
import torch.nn.functional as F

@MODELS.register_module()
class ThreeViewPillarFeatureNet(nn.Module):
    def __init__(self,
                 # BEV
                 bev_in_channels: int = 4,
                 bev_feat_channels: Tuple[int] = (64,),
                 # XOZ
                 xoz_in_channels: int = 4,
                 xoz_feat_channels: Tuple[int] = (32,),
                 # YOZ
                 yoz_in_channels: int = 4,
                 yoz_feat_channels: Tuple[int] = (32,),
                 # 公共
                 with_distance: bool = False,
                 with_cluster_center: bool = True,
                 with_voxel_center: bool = True,
                 bev_voxel_size: Tuple[float] = (0.32, 0.32, 4.0),
                 xoz_voxel_size: Tuple[float] = (0.32, 51.2, 0.125),
                 yoz_voxel_size: Tuple[float] = (51.2, 0.32, 0.125),
                 point_cloud_range: Tuple[float] = (0, -25.6, -3, 51.2, 25.6, 2),
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode: str = 'max',
                 legacy: bool = True,
                 # 融合模块参数
                 d_model: int = 128,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.point_cloud_range = point_cloud_range
        x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range

        # 网格尺寸
        self.H = int((y_max - y_min) / bev_voxel_size[1])   # BEV Y方向格子数
        self.W = int((x_max - x_min) / bev_voxel_size[0])   # BEV X方向格子数
        self.L = int((z_max - z_min) / xoz_voxel_size[2])   # XOZ Z方向格子数
        side_W = int((x_max - x_min) / xoz_voxel_size[0])
        assert self.W == side_W, "XOZ 与 BEV 的 X 方向格子数不一致"
        self.Y = int((y_max - y_min) / yoz_voxel_size[1])   # YOZ Y方向格子数
        yoz_L = int((z_max - z_min) / yoz_voxel_size[2])    # YOZ Z方向格子数
        assert self.L == yoz_L, "XOZ 与 YOZ 的 Z 方向格子数不一致"

        # 三个视图的 PFN
        self.bev_pfn = MODELS.build(dict(
            type='ViewPillarFeatureNet',
            view_type='bev',
            in_channels=bev_in_channels,
            feat_channels=bev_feat_channels,
            with_distance=with_distance,
            with_cluster_center=with_cluster_center,
            with_voxel_center=with_voxel_center,
            voxel_size=bev_voxel_size,
            point_cloud_range=point_cloud_range,
            norm_cfg=norm_cfg,
            mode=mode,
            legacy=legacy
        ))
        self.xoz_pfn = MODELS.build(dict(
            type='ViewPillarFeatureNet',
            view_type='side_xz',
            in_channels=xoz_in_channels,
            feat_channels=xoz_feat_channels,
            with_distance=with_distance,
            with_cluster_center=with_cluster_center,
            with_voxel_center=with_voxel_center,
            voxel_size=xoz_voxel_size,
            point_cloud_range=point_cloud_range,
            norm_cfg=norm_cfg,
            mode=mode,
            legacy=legacy
        ))
        self.yoz_pfn = MODELS.build(dict(
            type='ViewPillarFeatureNet',
            view_type='side_yz',          # 关键修改
            in_channels=yoz_in_channels,
            feat_channels=yoz_feat_channels,
            with_distance=with_distance,
            with_cluster_center=with_cluster_center,
            with_voxel_center=with_voxel_center,
            voxel_size=yoz_voxel_size,    # 应为 (51.2, 0.32, 0.125)
            point_cloud_range=point_cloud_range,
            norm_cfg=norm_cfg,
            mode=mode,
            legacy=legacy
        ))

        # 融合模块
        max_len = max(self.L, self.Y)
        self.fusion_xoz_yoz = MODELS.build(dict(
            type='SpatialChannelFusion',
            in_ch_bev=xoz_feat_channels[-1],
            in_ch_side=yoz_feat_channels[-1],
            out_ch=xoz_feat_channels[-1],
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_len=max_len
        ))
        self.fusion_bev_xoz = MODELS.build(dict(
            type='SpatialChannelFusion',
            in_ch_bev=bev_feat_channels[-1],
            in_ch_side=xoz_feat_channels[-1],
            out_ch=bev_feat_channels[-1],
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_len=max_len
        ))

    def forward(self,
                bev_features, bev_num_points, bev_coors,
                xoz_features, xoz_num_points, xoz_coors,
                yoz_features, yoz_num_points, yoz_coors):
        # 1. 得到 pillar 特征
        bev_pillar = self.bev_pfn(bev_features, bev_num_points, bev_coors)
        xoz_pillar = self.xoz_pfn(xoz_features, xoz_num_points, xoz_coors)
        yoz_pillar = self.yoz_pfn(yoz_features, yoz_num_points, yoz_coors)

        # 2. batch size
        batch_size = int(bev_coors[:, 0].max().item()) + 1
        device = bev_features.device

        # 3. 散列到特征图
        # BEV: (B, H, W, C_bev)
        bev_map = torch.zeros(batch_size, self.H, self.W,
                              bev_pillar.shape[-1], device=device)
        bev_y = bev_coors[:, 2].long()
        bev_x = bev_coors[:, 3].long()
        bev_map[bev_coors[:, 0].long(), bev_y, bev_x] = bev_pillar

        # XOZ: (B, L, W, C_xoz)
        xoz_map = torch.zeros(batch_size, self.L, self.W,
                              xoz_pillar.shape[-1], device=device)
        xoz_z = xoz_coors[:, 2].long()
        xoz_x = xoz_coors[:, 3].long()
        xoz_map[xoz_coors[:, 0].long(), xoz_z, xoz_x] = xoz_pillar

        # YOZ: (B, L, Y, C_yoz)
        yoz_map = torch.zeros(batch_size, self.L, self.Y, yoz_pillar.shape[-1], device=device)
        # 修正索引：根据体素化输出顺序，取第3列作为z索引，第2列作为y索引
        yoz_z = yoz_coors[:, 3].long()
        yoz_y = yoz_coors[:, 2].long()
        yoz_map[yoz_coors[:, 0].long(), yoz_z, yoz_y] = yoz_pillar

        # 4. 将 YOZ 沿 Y 方向池化到 W 尺寸，以便与 XOZ 融合
        # 注意：若 Y != W，则进行自适应池化
        if self.Y != self.W:
            # 将 (B, L, Y, C) -> (B, C, L, Y) -> 池化到 (B, C, L, W) -> (B, L, W, C)
            yoz_map_pooled = yoz_map.permute(0, 3, 1, 2)  # (B, C, L, Y)
            yoz_map_pooled = F.adaptive_avg_pool2d(yoz_map_pooled,
                                                   (self.L, self.W))
            yoz_map_pooled = yoz_map_pooled.permute(0, 2, 3, 1)  # (B, L, W, C)
        else:
            yoz_map_pooled = yoz_map

        # 5. 融合 XOZ 和 YOZ (逐样本)
        fused_xoz_list = []
        for b in range(batch_size):
            fused = self.fusion_xoz_yoz(xoz_map[b], yoz_map_pooled[b])
            fused_xoz_list.append(fused.unsqueeze(0))
        fused_xoz = torch.cat(fused_xoz_list, dim=0)  # (B, L, W, C_out)

        # 6. 融合 BEV 和 增强的 XOZ
        fused_bev_list = []
        for b in range(batch_size):
            fused = self.fusion_bev_xoz(bev_map[b], fused_xoz[b])
            fused_bev_list.append(fused.unsqueeze(0))
        fused_bev = torch.cat(fused_bev_list, dim=0)  # (B, H, W, C_out)

        return fused_bev


@MODELS.register_module()
class DynamicPillarFeatureNet(PillarFeatureNet):
    """Pillar Feature Net using dynamic voxelization.

    The network prepares the pillar features and performs forward pass
    through PFNLayers. The main difference is that it is used for
    dynamic voxels, which contains different number of points inside a voxel
    without limits.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels: Optional[int] = 4,
                 feat_channels: Optional[tuple] = (64, ),
                 with_distance: Optional[bool] = False,
                 with_cluster_center: Optional[bool] = True,
                 with_voxel_center: Optional[bool] = True,
                 voxel_size: Optional[Tuple[float]] = (0.2, 0.2, 4),
                 point_cloud_range: Optional[Tuple[float]] = (0, -40, -3, 70.4,
                                                              40, 1),
                 norm_cfg: Optional[dict] = dict(
                     type='BN1d', eps=1e-3, momentum=0.01),
                 mode: Optional[str] = 'max',
                 legacy: Optional[bool] = True):
        super(DynamicPillarFeatureNet, self).__init__(
            in_channels,
            feat_channels,
            with_distance,
            with_cluster_center=with_cluster_center,
            with_voxel_center=with_voxel_center,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            norm_cfg=norm_cfg,
            mode=mode,
            legacy=legacy)
        feat_channels = [self.in_channels] + list(feat_channels)
        pfn_layers = []
        # TODO: currently only support one PFNLayer

        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            pfn_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False), norm_layer,
                    nn.ReLU(inplace=True)))
        self.num_pfn = len(pfn_layers)
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.pfn_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        self.cluster_scatter = DynamicScatter(
            voxel_size, point_cloud_range, average_points=True)

    def map_voxel_center_to_point(self, pts_coors: Tensor, voxel_mean: Tensor,
                                  voxel_coors: Tensor) -> Tensor:
        """Map the centers of voxels to its corresponding points.

        Args:
            pts_coors (torch.Tensor): The coordinates of each points, shape
                (M, 3), where M is the number of points.
            voxel_mean (torch.Tensor): The mean or aggregated features of a
                voxel, shape (N, C), where N is the number of voxels.
            voxel_coors (torch.Tensor): The coordinates of each voxel.

        Returns:
            torch.Tensor: Corresponding voxel centers of each points, shape
                (M, C), where M is the number of points.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_y = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0] + 1
        canvas_len = canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_channel, canvas_len)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[:, indices.long()] = voxel_mean.t()

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        center_per_point = canvas[:, voxel_index.long()].t()
        return center_per_point

    def forward(self, features: Tensor, coors: Tensor) -> Tensor:
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel

        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        for i, pfn in enumerate(self.pfn_layers):
            point_feats = pfn(features)
            voxel_feats, voxel_coors = self.pfn_scatter(point_feats, coors)
            if i != len(self.pfn_layers) - 1:
                # need to concat voxel feats if it is not the last pfn
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        return voxel_feats, voxel_coors

import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor
from mmdet3d.registry import MODELS
from typing import Tuple, List, Optional

class CrossViewFusion(nn.Module):
    def __init__(self, in_ch_bev, in_ch_side, out_ch, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.proj_bev = nn.Linear(in_ch_bev, d_model)
        self.proj_side = nn.Linear(in_ch_side, d_model)
        self.pos_enc_bev = PositionalEncoding(d_model, max_len=200)
        self.pos_enc_side = PositionalEncoding(d_model, max_len=200)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, out_ch)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feat_bev, feat_side):
        # feat_bev: (B*W, H, C1)
        # feat_side: (B*W, L, C2)
        # 投影
        q = self.proj_bev(feat_bev)   # (B*W, H, d)
        k = self.proj_side(feat_side) # (B*W, L, d)
        v = k

        # 转置为 (seq, batch, dim)
        q = q.permute(1, 0, 2)   # (H, B*W, d)
        k = k.permute(1, 0, 2)   # (L, B*W, d)
        v = v.permute(1, 0, 2)   # (L, B*W, d)

        # 位置编码
        q = self.pos_enc_bev(q)
        k = self.pos_enc_side(k)
        v = self.pos_enc_side(v)

        B = q.size(1)   # B*W
        # 多头投影
        q = self.q_proj(q).view(q.size(0), B, self.n_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        k = self.k_proj(k).view(k.size(0), B, self.n_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        v = self.v_proj(v).view(v.size(0), B, self.n_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        # 现在形状: (B*W, n_heads, seq_len, head_dim)

        # 缩放点积注意力
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0
        )  # (B*W, n_heads, H, head_dim)

        # 合并头
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, self.d_model)  # (B*W, H, d)

        # 残差 + 归一化
        attn_output = self.norm(attn_output + q.permute(0, 2, 1, 3).reshape(B, -1, self.d_model))

        # 输出投影
        out = self.out_proj(attn_output)   # (B*W, H, out_ch)
        return out

class PositionalEncoding(nn.Module):
    """可学习位置编码"""
    def __init__(self, d_model, max_len=200):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, 1, d_model) * 0.02)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        seq_len = x.size(0)
        if seq_len > self.pe.size(0):
            # 动态扩展（可根据实际情况调整）
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.pe.size(0)}")
        return x + self.pe[:seq_len, :, :]

@MODELS.register_module()
class ViewSparseConvEncoder(nn.Module):
    """对单个视图（xoy 或 xoz）进行稀疏卷积编码，输出密集特征图。

    Args:
        in_channels (int): 每个点的输入特征维度。
        feat_channels (Tuple[int]): PFN 中间层通道数（用于点特征编码）。
        out_channels (int): 输出特征通道数。
        voxel_size (Tuple[float, float, float]): 体素大小 (vx, vy, vz)。
        point_cloud_range (Tuple[float, float, float, float, float, float]): 点云范围。
        grid_shape (Tuple[int, int]): 输出密集图的空间形状 (H, W) 或 (Z, X)。
        norm_cfg (dict): 归一化配置。
        mode (str): PFN 池化模式，默认 'max'。
        num_conv_layers (int): 稀疏卷积层数，默认 2。
        conv_channels (List[int]): 每层稀疏卷积的输出通道数，默认 [64, 64]。
        kernel_size (int): 卷积核大小，默认 3。
        stride (int): 卷积步长，默认 1。
        padding (int): 填充，默认 1。
    """
    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: Tuple[int] = (64,),
                 out_channels: int = 64,
                 voxel_size: Tuple[float, float, float] = (0.32, 0.32, 4.0),
                 point_cloud_range: Tuple[float] = (0, -25.6, -3, 51.2, 25.6, 2),
                 grid_shape: Tuple[int, int] = None,  # (H, W) or (Z, X)
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode: str = 'max',
                 num_conv_layers: int = 2,
                 conv_channels: List[int] = [64, 64],
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.grid_shape = grid_shape
        self.num_conv_layers = num_conv_layers
        self.conv_channels = conv_channels

        # PFN：对每个体素内的点进行编码
        self.pfn = PFNLayer(
            in_channels=in_channels,
            out_channels=feat_channels[0],
            norm_cfg=norm_cfg,
            last_layer=True,      # 因为是单层，设为最后一层
            mode=mode
        )
        # 如果有多个中间层，可以叠加，但一般一层就够了

        # 稀疏卷积层
        self.sparse_convs = nn.ModuleList()
        in_ch = feat_channels[0]
        for i in range(num_conv_layers):
            out_ch = conv_channels[i] if i < len(conv_channels) else out_channels
            self.sparse_convs.append(
                spconv.SparseSequential(
                    spconv.SubMConv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
            in_ch = out_ch

        # 最终输出投影（将 conv 输出映射到 out_channels）
        if in_ch != out_channels:
            self.proj = nn.Linear(in_ch, out_channels)
        else:
            self.proj = nn.Identity()

    def forward(self, voxels, num_points, coors, batch_size):
        # coors: (N, 3) -> [batch, spatial_dim0, spatial_dim1]
        # spatial_dim0 对应 grid_shape[1]（x方向），spatial_dim1 对应 grid_shape[0]（y或z方向）
        voxel_features = self.pfn(voxels, num_points).squeeze(1)   # (N, C)

        # 断言坐标在网格范围内
        spatial_coors = coors[:, 1:]   # (N, 2)
        # 打印 coors 各列的最小最大值
        # print("coors[:, 0] min/max:", coors[:, 0].min(), coors[:, 0].max())
        # print("coors[:, 1] min/max:", coors[:, 1].min(), coors[:, 1].max())
        # print("coors[:, 2] min/max:", coors[:, 2].min(), coors[:, 2].max())
        assert (spatial_coors[:, 0] < self.grid_shape[1]).all(), "x out of range"
        assert (spatial_coors[:, 1] < self.grid_shape[0]).all(), "y/z out of range"

        # 构建稀疏张量（包含batch列）
        indices = coors.int()   # (N, 3)
        sparse_tensor = SparseConvTensor(
            features=voxel_features,
            indices=indices,
            spatial_shape=self.grid_shape[::-1],   # (W, H) 或 (W, L)
            batch_size=batch_size
        )

        # 稀疏卷积...
        x = sparse_tensor
        for conv in self.sparse_convs:
            x = conv(x)

        x = x.replace_feature(self.proj(x.features))
        dense = x.dense()                     # (batch, C, W, H) 或 (batch, C, W, L)
        dense = dense.permute(0, 3, 2, 1)     # (batch, H, W, C) 或 (batch, L, W, C)
        return dense

@MODELS.register_module()
class MultiViewSparseConvEncoder(nn.Module):
    """多视图稀疏卷积编码器，支持 xoy 和 xoz 平面，并通过交叉注意力融合。

    Args:
        bev_voxel_size (Tuple[float, float, float]): BEV 体素大小 (vx, vy, vz)。
        side_voxel_size (Tuple[float, float, float]): 侧视图体素大小 (vx, vy, vz)。
        point_cloud_range (Tuple[float]): 点云范围。
        bev_in_channels (int): BEV 每个点的输入特征维度。
        side_in_channels (int): 侧视图每个点的输入特征维度。
        bev_feat_channels (Tuple[int]): BEV PFN 中间层通道数。
        side_feat_channels (Tuple[int]): 侧视图 PFN 中间层通道数。
        bev_out_channels (int): BEV 输出通道数。
        side_out_channels (int): 侧视图输出通道数。
        bev_num_conv_layers (int): BEV 稀疏卷积层数。
        side_num_conv_layers (int): 侧视图稀疏卷积层数。
        conv_channels (List[int]): 稀疏卷积通道数列表。
        d_model (int): Transformer 中间维度。
        n_heads (int): 注意力头数。
        dropout (float): Dropout 概率。
        norm_cfg (dict): 归一化配置。
        mode (str): PFN 池化模式。
    """
    def __init__(self,
                 bev_voxel_size: Tuple[float, float, float] = (0.32, 0.32, 4.0),
                 side_voxel_size: Tuple[float, float, float] = (0.32, 51.2, 0.125),
                 point_cloud_range: Tuple[float] = (0, -25.6, -3, 51.2, 25.6, 2),
                 bev_in_channels: int = 5,
                 side_in_channels: int = 5,
                 bev_feat_channels: Tuple[int] = (64,),
                 side_feat_channels: Tuple[int] = (32,),
                 bev_out_channels: int = 64,
                 side_out_channels: int = 32,
                 bev_num_conv_layers: int = 2,
                 side_num_conv_layers: int = 2,
                 conv_channels: List[int] = [64, 64],
                 d_model: int = 128,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode: str = 'max'):
        super().__init__()
        self.point_cloud_range = point_cloud_range
        self.bev_out_channels = bev_out_channels
        self.side_out_channels = side_out_channels
        x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range

        # 计算各视图的网格尺寸
        bev_vx, bev_vy, bev_vz = bev_voxel_size
        self.H = int((y_max - y_min) / bev_vy)  # y 方向格子数
        self.W = int((x_max - x_min) / bev_vx)  # x 方向格子数

        side_vx, side_vy, side_vz = side_voxel_size
        self.L = int((z_max - z_min) / side_vz)  # z 方向格子数
        # 侧视图的 x 方向格子数必须与 BEV 一致
        side_W = int((x_max - x_min) / side_vx)
        assert self.W == side_W, f"X 维度格子数不一致: BEV={self.W}, side={side_W}"

        # BEV 编码器
        self.bev_encoder = ViewSparseConvEncoder(
            in_channels=bev_in_channels,
            feat_channels=bev_feat_channels,
            out_channels=bev_out_channels,
            voxel_size=bev_voxel_size,
            point_cloud_range=point_cloud_range,
            grid_shape=(self.H, self.W),
            norm_cfg=norm_cfg,
            mode=mode,
            num_conv_layers=bev_num_conv_layers,
            conv_channels=conv_channels[:bev_num_conv_layers]
        )

        # 侧视图编码器
        self.side_encoder = ViewSparseConvEncoder(
            in_channels=side_in_channels,
            feat_channels=side_feat_channels,
            out_channels=side_out_channels,
            voxel_size=side_voxel_size,
            point_cloud_range=point_cloud_range,
            grid_shape=(self.L, self.W),
            norm_cfg=norm_cfg,
            mode=mode,
            num_conv_layers=side_num_conv_layers,
            conv_channels=conv_channels[:side_num_conv_layers]
        )

        # 融合模块（基于 x 轴共享）
        self.fusion = CrossViewFusion(
            in_ch_bev=bev_out_channels,
            in_ch_side=side_out_channels,
            out_ch=bev_out_channels,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )

    def forward(self, bev_voxels, bev_num_points, bev_coors,
                side_voxels, side_num_points, side_coors):
        batch_size = int(bev_coors[:, 0].max().item()) + 1

        # BEV：需要 [batch, x, y]
        # 假设原始 bev_coors 格式为 [batch, x, y, z]
        bev_coors_spatial = torch.cat([
            bev_coors[:, 0:1],      # batch
            bev_coors[:, 3:4],      # x
            bev_coors[:, 2:3]       # y
        ], dim=1)

        side_coors_spatial = torch.cat([
            side_coors[:, 0:1],     # batch
            side_coors[:, 3:4],     # x
            side_coors[:, 1:2]      # z
        ], dim=1)

        # 分别编码
        bev_map = self.bev_encoder(bev_voxels, bev_num_points, bev_coors_spatial, batch_size)   # (B, H, W, Cb)
        side_map = self.side_encoder(side_voxels, side_num_points, side_coors_spatial, batch_size) # (B, L, W, Cs)

        # 转换为以 W 为公共维度
        bev_perm = bev_map.permute(0, 2, 1, 3)   # (B, W, H, C_bev)
        side_perm = side_map.permute(0, 2, 1, 3) # (B, W, L, C_side)

        B, W, H, Cb = bev_perm.shape
        _, _, L, Cs = side_perm.shape

        # 断言确保维度一致
        assert W == side_perm.shape[1], f"W mismatch: {W} vs {side_perm.shape[1]}"
        assert Cb == self.bev_out_channels, f"Cb {Cb} vs {self.bev_out_channels}"
        assert Cs == self.side_out_channels, f"Cs {Cs} vs {self.side_out_channels}"

        # 融合时，将每个 batch 的每个 x 位置作为一个序列
        bev_seq = bev_perm.contiguous().reshape(B * W, H, Cb)   # (B*W, H, Cb)
        side_seq = side_perm.contiguous().reshape(B * W, L, Cs) # (B*W, L, Cs)

        # 可选：打印形状用于调试
        # print(f"bev_seq shape: {bev_seq.shape}, side_seq shape: {side_seq.shape}")

        fused_seq = self.fusion(bev_seq, side_seq)  # (B*W, H, Cb)

        # 恢复形状
        fused_perm = fused_seq.reshape(B, W, H, Cb)   # (B, W, H, Cb)
        fused = fused_perm.permute(0, 2, 1, 3)       # (B, H, W, Cb)
        return fused

