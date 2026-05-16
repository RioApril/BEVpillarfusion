_base_ = [
    './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_vod-3d.py']

point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]  # 与点云分支保持一致
input_modality = dict(use_lidar=True, use_camera=True)
voxel_size = [0.32, 0.32, 0.125]
bev_h = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])  # Y 方向网格数
bev_w = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])  # X 方向网格数
backend_args = None

# === BGR order (as read by OpenCV) ===
# mean = [146.372, 136.565, 107.780]
# std  = [75.099, 74.745, 70.402]

# === RGB order (converted) ===
# mean = [107.780, 136.565, 146.372]
# std  = [70.402, 74.745, 75.099]

model = dict(
    type='MultiViewBEVFusion',   # 复用点云分支的模型类
    data_preprocessor=dict(
        type='MultiViewDataPreprocessor',  # 同时支持图像和双视图体素
        # 图像预处理参数
        # mean=[123.675, 116.28, 103.53],
        # std=[58.395, 57.12, 57.375],
        mean = [107.780, 136.565, 146.372],
        std  = [70.402, 74.745, 75.099],
        bgr_to_rgb=True
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
    # view_transform=dict(
    #     type='DepthLSSTransform',
    #     in_channels=256,
    #     out_channels=64,
    #     image_size=[256, 704],
    #     feature_size=[32, 88],
    #     xbound=[point_cloud_range[0], point_cloud_range[3], 0.32],
    #     ybound=[point_cloud_range[1], point_cloud_range[4], 0.32],
    #     zbound=[-10.0, 10.0, 20.0],
    #     dbound=[1.0, 60.0, 0.5],
    #     downsample=2),
    # 融合层
    # fusion_layer=dict(
    #     type='ConvFuser', in_channels=[64, 64], out_channels=64,
    #     init_mode='img_direct'))
    view_transform=None,   # 不再使用 LSS，设置为恒等变换
    fusion_layer=dict(
        type='BEVPillarFusion',
        image_channels=256,                # 图像特征通道数（来自 img_neck 的 out_channels）
        lidar_bev_channels=64,            # 点云 BEV 特征通道数（需要根据点云分支实际输出调整）
        out_channels=64,                  # 融合后的输出通道数
        bev_h=bev_h,
        bev_w=bev_w,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        num_samples=8,
        attn_hidden_dim=32,
        img_downsample_channels=32,
        mode='normal'
        )
    # fusion_layer=dict(
    #     type='YOZPillarFusion',
    #     image_channels=256,                # 图像特征通道数（来自 img_neck 的 out_channels）
    #     lidar_bev_channels=64,            # 点云 BEV 特征通道数（需要根据点云分支实际输出调整）
    #     out_channels=64,                  # 融合后的输出通道数
    #     bev_h=bev_h,
    #     bev_w=bev_w,
    #     point_cloud_range=point_cloud_range,
    #     voxel_size=voxel_size,
    #     num_samples=8,
    #     attn_hidden_dim=32,
    #     img_downsample_channels=64,
    #     )
    )

# epochs = 6
# lr = 0.0002

epochs = 10
lr = 0.0005

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
    dict(type='CosineAnnealingMomentum', eta_min=0.85, begin=0, end=epochs*0.4,
         by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', eta_min=1, begin=epochs*0.4, end=epochs,
         by_epoch=True, convert_to_iter_based=True)
]

auto_scale_lr = dict(enable=False, base_batch_size=32)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1))
