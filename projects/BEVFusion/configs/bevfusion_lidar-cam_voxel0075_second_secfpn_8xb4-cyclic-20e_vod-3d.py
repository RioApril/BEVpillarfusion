_base_ = [
    './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_vod-3d.py']

point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]  # 与点云分支保持一致
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

model = dict(
    type='MultiViewBEVFusion',   # 复用点云分支的模型类
    data_preprocessor=dict(
        type='MultiViewDataPreprocessor',  # 同时支持图像和双视图体素
        # 图像预处理参数
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False
        # 双视图体素层（与点云分支一致）
        ),
    # 图像骨干网络
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth')),
    # 图像颈部网络
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
    # 视图变换 (LSS)
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=64,
        image_size=[256, 408],
        feature_size=[32, 51],
        xbound=[point_cloud_range[0], point_cloud_range[3], 0.32],
        ybound=[point_cloud_range[1], point_cloud_range[4], 0.32],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2),
    # 融合层
    fusion_layer=dict(
        type='ConvFuser', in_channels=[64, 64], out_channels=64,
        init_mode='noraml'))

epochs = 10
lr = 0.0003

# ========== 训练配置 ==========
train_cfg = dict(by_epoch=True, max_epochs=epochs, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# 优化器和调度器（沿用融合配置中的设置）
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333333, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=0, T_max=epochs, end=epochs, by_epoch=True,
         eta_min_ratio=1e-4, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', eta_min=0.85/0.95, begin=0, end=epochs*0.4,
         by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', eta_min=1, begin=epochs*0.4, end=epochs,
         by_epoch=True, convert_to_iter_based=True)
]

auto_scale_lr = dict(enable=False, base_batch_size=32)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1))