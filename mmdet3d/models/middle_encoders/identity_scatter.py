import torch
import torch.nn as nn
from mmdet3d.registry import MODELS

@MODELS.register_module()
class IdentityMiddleEncoder(nn.Module):
    """接收已散列的 BEV 特征图 (batch, H, W, C)，转换为 (batch, C, H, W) 供 backbone 使用。"""
    def __init__(self, in_channels=None, output_shape=None):
        super().__init__()
        self.in_channels = in_channels
        self.output_shape = output_shape

    def forward(self, voxel_features, coors=None, batch_size=None):
        # voxel_features: (batch, H, W, C)
        # 转换为 (batch, C, H, W)
        # print(f"[IdentityMiddleEncoder] input shape: {voxel_features.shape}")
        # if voxel_features.shape[-1] == 64 and voxel_features.shape[1] == 160:
        #     # 第二次调用时打印调用栈
        #     traceback.print_stack()
        out = voxel_features.permute(0, 3, 1, 2).contiguous()
        # print(f"[IdentityMiddleEncoder] output shape: {out.shape}")
        return out