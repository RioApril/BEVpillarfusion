_base_ = ['/home/vipuser/project/mmdetection3d/projects/BEVFusion/paper/default_runtime.py',
          '/home/vipuser/project/mmdetection3d/projects/BEVFusion/paper/dataset/vod-3d-3class_img.py'
          ]

custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion'],  # 导入自定义模型类
    allow_failed_imports=False)

# ========== 数据集参数 ==========
voxel_size = [0.32, 0.32, 0.125]
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
dataset_type = 'VodDataset'
data_root = 'data/view_of_delft_PUBLIC/radar_3frames/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)
backend_args = None

# 计算网格尺寸
nx = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])  # 160
ny = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])  # 160
nz = int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2])  # 40
grid_size = [nx, ny, nz]   # [160, 160, 40]

# ========== 模型配置 ==========
model = dict(
    type='MultiViewBEVFusion',
    data_preprocessor=dict(
        type='MultiViewDataPreprocessor',
        bev_voxel_layer=dict(
            voxel_size=[0.32, 0.32, 5.0],
            point_cloud_range=point_cloud_range,
            max_num_points=32,
            max_voxels=(16000, 40000)),
        side_voxel_layer=dict(
            voxel_size=[0.32, 51.2, 0.125],
            point_cloud_range=point_cloud_range,
            max_num_points=32,
            max_voxels=(16000, 40000))),
    # 点云编码器：双视图 Pillar Feature Net
    pts_voxel_encoder=dict(
        type='MultiViewPillarFeatureNet',
        bev_in_channels=5,
        bev_feat_channels=[64],
        side_in_channels=5,
        side_feat_channels=[32],
        with_distance=False,
        with_cluster_center=False,
        with_voxel_center=False,
        with_convolution=False,
        conv_kernel_size=7,
        bev_voxel_size=[0.32, 0.32, 5],
        side_voxel_size=[0.32, 51.2, 0.125],
        point_cloud_range=point_cloud_range,
        mode='fused'
    ),
    # 中间编码器：仅做维度转换 (B,H,W,C) -> (B,C,H,W)
    pts_middle_encoder=dict(
        type='IdentityMiddleEncoder',
        in_channels=64,
        output_shape=[ny, nx]),  # H=160, W=160
    # 2D 骨干网络
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    # 颈部网络
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [0,-25.6,-0.6,51.2,25.6,-0.6],
                [0,-25.6,-0.6,51.2,25.6,-0.6],
                [0, -25.6, -1.78, 51.2, 25.6, -1.78],
            ],
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50)
        )


lr = 0.001  # 保持官方基准
epoch_num = 30  # 【修改】总轮次设为 60
# ========== 优化器和调度器 ==========
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.4,
        eta_min=lr * 10,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.6,
        eta_min=lr * 1e-4,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.6,
        eta_min=1,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        convert_to_iter_based=True)
]

train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=2)
val_cfg = dict()
test_cfg = dict()
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, by_epoch=True)
)

# load_from = '/home/vipuser/project/mmdetection3d/work_dirs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_vod-3d/epoch_40.pth'  # 【修改】预训练权重路径