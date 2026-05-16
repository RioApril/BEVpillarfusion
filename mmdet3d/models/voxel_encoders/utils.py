# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmcv.cnn import build_norm_layer
from torch import Tensor, nn
from torch.nn import functional as F


def get_paddings_indicator(actual_num: Tensor,
                           max_num: Tensor,
                           axis: int = 0) -> Tensor:
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel

    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class VFELayer(nn.Module):
    """Voxel Feature Encoder layer.

    The voxel encoder is composed of a series of these layers.
    This module do not support average pooling and only support to use
    max pooling to gather features inside a VFE.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
        max_out (bool): Whether aggregate the features of points inside
            each voxel and only return voxel features.
        cat_max (bool): Whether concatenate the aggregated features
            and pointwise features.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: Optional[dict] = dict(
                     type='BN1d', eps=1e-3, momentum=0.01),
                 max_out: Optional[bool] = True,
                 cat_max: Optional[bool] = True):
        super(VFELayer, self).__init__()
        self.cat_max = cat_max
        self.max_out = max_out
        # self.units = int(out_channels / 2)

        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward function.

        Args:
            inputs (torch.Tensor): Voxels features of shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.

        Returns:
            torch.Tensor: Voxel features. There are three mode under which the
                features have different meaning.
                - `max_out=False`: Return point-wise features in
                    shape (N, M, C).
                - `max_out=True` and `cat_max=False`: Return aggregated
                    voxel features in shape (N, C)
                - `max_out=True` and `cat_max=True`: Return concatenated
                    point-wise features in shape (N, M, C).
        """
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        voxel_count = inputs.shape[1]

        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        pointwise = F.relu(x)
        # [K, T, units]
        if self.max_out:
            aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        else:
            # this is for fusion layer
            return pointwise

        if not self.cat_max:
            return aggregated.squeeze(1)
        else:
            # [K, 1, units]
            repeated = aggregated.repeat(1, voxel_count, 1)
            concatenated = torch.cat([pointwise, repeated], dim=2)
            # [K, T, 2 * units]
            return concatenated


class PFNLayer(nn.Module):
    """Pillar Feature Net Layer with Adaptive Max-Attention Fusion.

    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.

    Modified to support adaptive fusion of max pooling and multi-head attention,
    improving performance on both large objects (cars) and small objects (pedestrians).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict, optional): Config dict of normalization layers.
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        last_layer (bool, optional): If last_layer, there is no
            concatenation of features. Defaults to False.
        mode (str, optional): Pooling model to gather features inside voxels.
            Supported: 'max', 'avg', 'multihead_attn', 'fused'. 
            Defaults to 'fused' for adaptive fusion.
        attn_cfg (dict, optional): Configuration for multi-head attention.
            Required if mode='multihead_attn' or 'fused'. Should contain:
                - num_heads (int): Number of attention heads.
                - head_dim (int): Dimension per head.
            If not provided, defaults to num_heads=8, head_dim=out_channels//8
            (must be divisible).
        fuse_cfg (dict, optional): Configuration for fusion gating.
            Only used when mode='fused'. Should contain:
                - gate_channels (int): Internal channels for gate MLP.
                    Defaults to 32.
                - use_density_prior (bool): Whether to use num_voxels as 
                    prior for gating. Defaults to True.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: Optional[dict] = dict(
                     type='BN1d', eps=1e-3, momentum=0.01),
                 last_layer: Optional[bool] = False,
                 mode: Optional[str] = 'fused',
                 attn_cfg: Optional[dict] = None,
                 fuse_cfg: Optional[dict] = None):
        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        self.mode = mode

        # 计算中间通道数
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        # 基础线性层和归一化
        self.norm = build_norm_layer(norm_cfg, self.units)[1]
        self.linear = nn.Linear(in_channels, self.units, bias=False)

        # 根据模式初始化额外组件
        if self.mode in ['multihead_attn', 'fused']:
            # 解析注意力配置
            if attn_cfg is None:
                # 默认配置：8 个头，每个头维度为 units // 8
                num_heads = 8
                if self.units % num_heads != 0:
                    raise ValueError(f"units ({self.units}) must be divisible by num_heads ({num_heads})")
                head_dim = self.units // num_heads
            else:
                num_heads = attn_cfg.get('num_heads', 8)
                head_dim = attn_cfg.get('head_dim', self.units // num_heads)
                if num_heads * head_dim != self.units:
                    raise ValueError(f"num_heads * head_dim ({num_heads * head_dim}) != units ({self.units})")

            self.num_heads = num_heads
            self.head_dim = head_dim

            # 可学习的查询向量 (num_heads, head_dim)
            self.query = nn.Parameter(torch.zeros(num_heads, head_dim))
            nn.init.xavier_uniform_(self.query)

            # 键和值的投影层
            self.key_proj = nn.Linear(self.units, num_heads * head_dim, bias=False)
            self.value_proj = nn.Linear(self.units, num_heads * head_dim, bias=False)

            # 融合门控网络 (仅在 fused 模式下使用)
            if self.mode == 'fused':
                if fuse_cfg is None:
                    gate_channels = 32
                    use_density_prior = True
                else:
                    gate_channels = fuse_cfg.get('gate_channels', 32)
                    use_density_prior = fuse_cfg.get('use_density_prior', True)
                
                self.use_density_prior = use_density_prior
                # 门控输入维度：全局特征 (1) + 密度先验 (1, 可选)
                gate_input_dim = 1 + (1 if use_density_prior else 0)
                
                self.gate_mlp = nn.Sequential(
                    nn.Linear(gate_input_dim, gate_channels),
                    nn.ReLU(),
                    nn.Linear(gate_channels, 1),
                    nn.Sigmoid()
                )
        elif self.mode not in ['max', 'avg']:
            raise ValueError(f"Unsupported mode: {mode}. Supported: 'max', 'avg', 'multihead_attn', 'fused'")

    def forward(self,
                inputs: Tensor,
                num_voxels: Optional[Tensor] = None,
                aligned_distance: Optional[Tensor] = None) -> Tensor:
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (torch.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (torch.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.

        Returns:
            torch.Tensor: Features of Pillars.
        """
        # 共享的线性变换 + 归一化 + 激活
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)

        # 如果提供了 aligned_distance，则按原逻辑加权
        if aligned_distance is not None:
            x_geo = x.mul(aligned_distance.unsqueeze(-1))
        else:
            x_geo = x

        # 分支 A: 最大池化 (始终计算，用于融合或单独使用)
        x_max = torch.max(x_geo, dim=1, keepdim=True)[0]

        if self.mode == 'max':
            x_pooled = x_max
        elif self.mode == 'avg':
            # 注意：当 aligned_distance 存在时，这里已经乘以距离，所以求和后再除以实际点数
            x_sum = x_geo.sum(dim=1, keepdim=True)
            if num_voxels is not None:
                # 避免除以零（实际点数至少为 1，但保险起见）
                divisor = num_voxels.type_as(x).view(-1, 1, 1)
                x_pooled = x_sum / divisor.clamp(min=1)
            else:
                x_pooled = x_sum / x.size(1)  # 使用最大点数（可能包含填充点）
        elif self.mode in ['multihead_attn', 'fused']:
            # 生成掩码 (N, M)，标记有效点
            N, M, _ = x.shape
            if num_voxels is not None:
                # 每个 voxel 前 num_voxels[i] 个点为有效
                mask = torch.arange(M, device=x.device).unsqueeze(0) < num_voxels.unsqueeze(1)  # (N, M)
            else:
                mask = torch.ones(N, M, dtype=torch.bool, device=x.device)

            # 将 x 投影到键和值
            keys = self.key_proj(x)          # (N, M, num_heads * head_dim)
            values = self.value_proj(x)      # (N, M, num_heads * head_dim)

            # 重塑为 (N, M, num_heads, head_dim)
            keys = keys.view(N, M, self.num_heads, self.head_dim)
            values = values.view(N, M, self.num_heads, self.head_dim)

            # 扩展查询向量为 (N, num_heads, head_dim)
            query = self.query.unsqueeze(0).expand(N, -1, -1)  # (N, num_heads, head_dim)

            # 计算注意力分数：query 与每个点的 key 的点积 (N, M, num_heads)
            scores = torch.einsum('nhd,nmhd->nmh', query, keys)  # (N, M, num_heads)

            # 应用掩码：将无效位置的分数设为 -inf
            mask_expanded = mask.unsqueeze(-1)  # (N, M, 1)
            scores = scores.masked_fill(~mask_expanded, -float('inf'))

            # 计算注意力权重 (softmax 沿 M 维)
            attn_weights = F.softmax(scores, dim=1)  # (N, M, num_heads)

            # 加权求和得到每个头的输出 (N, num_heads, head_dim)
            x_attn = torch.einsum('nmh,nmhd->nhd', attn_weights, values)  # (N, num_heads, head_dim)

            # 展平并增加序列维度 (N, 1, units)
            x_attn = x_attn.reshape(N, 1, self.num_heads * self.head_dim)  # (N, 1, units)

            if self.mode == 'multihead_attn':
                x_pooled = x_attn
            elif self.mode == 'fused':
                # --- 门控融合逻辑 ---
                # 计算门控权重 alpha
                # 策略：使用全局平均特征 + 归一化点数 作为门控输入
                
                if self.use_density_prior and num_voxels is not None:
                    # 归一化点数到 0-1 (假设最大点数为 100，根据实际数据集调整)
                    # 可根据数据集最大点数调整分母 (如 64, 100 等)
                    density = (num_voxels / 48).view(-1, 1).clamp(0, 1)
                else:
                    density = None
                
                # 全局特征统计 (使用 Max 特征代表全局)
                global_feat = x_max.mean(dim=2, keepdim=True)  # (N, 1, 1)
                
            if density is not None:
                density = density.unsqueeze(-1)  # (N, 1, 1)
                gate_input = torch.cat([global_feat, density], dim=2)  # (N, 1, 2)
            else:
                gate_input = global_feat  # (N, 1, 1)
                
            alpha = self.gate_mlp(gate_input)  # (N, 1, 1)
                
            # 融合：alpha * Attn + (1 - alpha) * Max
            # 当 alpha 接近 1 (点数少) -> 多用 Attention (适合行人)
            # 当 alpha 接近 0 (点数多) -> 多用 Max (适合汽车)
            x_pooled = alpha * x_attn + (1 - alpha) * x_max
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        if self.last_vfe:
            return x_pooled
        else:
            # 将池化结果重复到每个点，并与原特征拼接
            x_repeat = x_pooled.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated
