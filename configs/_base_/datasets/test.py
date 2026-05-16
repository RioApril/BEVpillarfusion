# dataset settings
dataset_type = 'VodDataset'
data_root = 'data/view_of_delft_PUBLIC/radar_3frames/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]

# 开启相机模态
input_modality = dict(use_lidar=True, use_camera=True)
metainfo = dict(classes=class_names)
backend_args = None

# 可选的数据库采样（若需要请保留，否则可注释）
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=class_names,
    sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,          # 改为4维（与mvxnet一致）
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

# ========== 训练 pipeline（参考 mvxnet 添加图像） ==========
train_pipeline = [
    # 1. 加载点云（改用4维）
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,          # x, y, z, intensity
        use_dim=4,
        backend_args=backend_args),
    # 2. 加载图像
    dict(type='LoadImageFromFile', backend_args=backend_args),
    # 3. 加载3D标注（同时会生成2D标注框，如果数据中有的话）
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # 4. 可选：数据库采样（需确保图像也能正确同步，复杂场景下建议注释）
    # dict(type='ObjectSample', db_sampler=db_sampler),
    # 5. 图像尺寸随机调整（保持宽高比）
    dict(
        type='RandomResize',
        scale=[(640, 192), (2560, 768)],   # 与mvxnet相同
        keep_ratio=True),
    # 6. 点云全局旋转/缩放/平移（不影响图像）
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.2, 0.2, 0.2]),
    # 7. 随机水平翻转（同时翻转点云和图像）
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    # 8. 点云范围过滤
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    # 9. 打包输入（添加图像和2D标签）
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d',
            'gt_bboxes', 'gt_labels'   # 2D框和标签（由LoadAnnotations3D自动生成）
        ])
]

# ========== 测试 pipeline ==========
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1280, 384),     # 与mvxnet测试尺度一致
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # 图像resize（固定尺寸）
            dict(type='Resize', scale=0, keep_ratio=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]

# 评估 pipeline（仅用于可视化，不包含图像也可，但保持与测试一致）
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]

# ========== 数据加载器 ==========
train_dataloader = dict(
    batch_size=10,
    num_workers=7,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(
                pts='training/velodyne_reduced',
                img='training/image_2'      # 根据VOD实际图像路径修改
            ),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR',
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            pts='training/velodyne_reduced',
            img='training/image_2'),
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = val_dataloader   # 测试与验证共用

# ========== 评估器（保持VOD原有度量） ==========
val_evaluator = dict(
    type='VODMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

# ========== 可视化 ==========
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')