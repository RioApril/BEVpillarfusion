voxel_size = [0.32, 0.32, 0.125]

model = dict(
    type='MultiViewVoxelNet',
    data_preprocessor=dict(
        type='MultiViewDataPreprocessor',
    bev_voxel_layer=dict(
        voxel_size=[0.32, 0.32, 4.0],
        point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2],
        max_num_points=32,
        max_voxels=(16000, 40000)
    ),
    side_voxel_layer=dict(
        voxel_size=[0.32, 51.2, 0.125],
        point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2],
        max_num_points=32,
        max_voxels=(16000, 40000)
    )),
    voxel_encoder=dict(
        type='MultiViewPillarFeatureNet',
        bev_in_channels=5,
        bev_feat_channels=[64],
        side_in_channels=5,
        side_feat_channels=[32],
        with_distance=False,
        with_cluster_center=False,
        with_voxel_center=False,
        bev_voxel_size=[0.32, 0.32, 5],
        side_voxel_size=[0.32, 51.2, 0.125],
        point_cloud_range=[0,-25.6,-3,51.2,25.6,2],
        ),
    middle_encoder=dict(
        type='IdentityMiddleEncoder', in_channels=64, output_shape=[160,160]),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    bbox_head=dict(
        type='CenterHead',
        in_channels=sum([128, 128, 128]),
        tasks=[
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist']),
            dict(num_class=1, class_names=['Car']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[0,-25.6,-3,51.2,25.6,2],
            max_num=500,
            pc_range=[0, -25.6],
            score_threshold=0.1,
            out_size_factor=2,
            voxel_size=voxel_size[:2],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),

    train_cfg=dict(
        grid_size = [160, 160, 1],
        point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2],
        voxel_size=voxel_size,
        out_size_factor=2,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        pts=dict(
            grid_size = [160, 160, 1],
            voxel_size=voxel_size,
            out_size_factor=2,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
            post_center_limit_range=[0,-25.6,-3,51.2,25.6,2],
            point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2],
            max_per_img=500,
            max_pool_nms=False,
            min_radius = [4, 0.85, 0.175],
            score_threshold=0.1,
            pc_range=[0, -25.6, -3, 51.2],
            out_size_factor=2,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
        pts=dict(
            post_center_limit_range=[0,-25.6,-3,51.2,25.6,2],
            max_per_img=500,
            max_pool_nms=False,
            min_radius = [4, 0.85, 0.175],
            score_threshold=0.1,
            pc_range=[0, -25.6, -3, 51.2],
            out_size_factor=2,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))