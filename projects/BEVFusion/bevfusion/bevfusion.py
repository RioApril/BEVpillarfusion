from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.models.detectors.single_stage import SingleStage3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization


@MODELS.register_module()
class BEVFusion(Base3DDetector):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)

        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

        # print(f"[DEBUG] pts_middle_encoder type: {type(self.pts_middle_encoder)}")
        # print(f"[DEBUG] pts_middle_encoder module: {self.pts_middle_encoder}")

        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None

        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)

        self.bbox_head = MODELS.build(bbox_head)

        self.init_weights()

        # print(self.pts_backbone.blocks)

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        return x

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature = self.extract_img_feat(imgs, deepcopy(points),
                                                lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix,
                                                lidar_aug_matrix,
                                                batch_input_metas)
            features.append(img_feature)
        pts_feature = self.extract_pts_feat(batch_inputs_dict)
        features.append(pts_feature)

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x) # FIX

        return x

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        losses = dict()
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)

        losses.update(bbox_loss)

        return losses

@MODELS.register_module()
class MultiViewBEVFusion(SingleStage3DDetector):
    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        mode: Optional[str] = 'mid',
        init_cfg: OptMultiConfig = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None
    ) -> None:
        # voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        # self.pts_voxel_layer = Voxelization(**voxelize_cfg)

        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)
        
        self.mode = mode

        # print(f"[DEBUG] pts_middle_encoder type: {type(self.pts_middle_encoder)}")
        # print(f"[DEBUG] pts_middle_encoder module: {self.pts_middle_encoder}")

        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None

        self.init_weights()

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()
    
        # print(self.pts_backbone.blocks)
    """BEVFusion with multi-view (BEV + side) point cloud encoder."""
    def extract_pts_feat(self, 
                         batch_inputs_dict: Optional[Dict[str, Tensor]] = None) -> Sequence[Tensor]:
        """Extract features from multi-view voxels."""
        # 获取双视图体素数据（由 MultiViewDataPreprocessor 产生）
        bev_voxels = batch_inputs_dict['bev_voxels']
        bev_num_points = batch_inputs_dict['bev_num_points']
        bev_coors = batch_inputs_dict['bev_coors']
        side_voxels = batch_inputs_dict['side_voxels']
        side_num_points = batch_inputs_dict['side_num_points']
        side_coors = batch_inputs_dict['side_coors']

        # 直接调用您的双视图编码器（融合了 BEV 和侧视图）
        # 该编码器内部已完成：PFN → scatter → 卷积 → 融合 → 输出 BEV 特征图
        x = self.pts_voxel_encoder(
            bev_voxels, bev_num_points, bev_coors,
            side_voxels, side_num_points, side_coors)
        batch_size = bev_coors[-1, 0].item() + 1

        # 将 (B, H, W, C) 转为 (B, C, H, W)
        x = self.pts_middle_encoder(x, bev_coors, batch_size)


        # print(f"pts_backbone id: {id(self.pts_backbone)}")
        # print(f"pts_neck id: {id(self.pts_neck)}")

        # 通过 backbone 和 neck
        # x = self.pts_backbone(x)
        # # print(f"After backbone, x type: {type(x)}")

        # x = self.pts_neck(x)
        return x
    
    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()
        
        debug_img_feat = False
        # ========== 调试：保存输入图像并检查 ==========
        if debug_img_feat:
            print(f"[DEBUG] B={B}, N={N}, C={C}, H={H}, W={W}")
            # 仅在主进程执行，避免多卡重复
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                # 获取数据预处理器中的 mean/std
                if hasattr(self, 'data_preprocessor'):
                    mean = self.data_preprocessor.mean
                    std = self.data_preprocessor.std
                    if torch.is_tensor(mean):
                        mean = mean.cpu().numpy()
                        std = std.cpu().numpy()
                else:
                    # 硬编码，与你配置中的 mean/std 保持一致（RGB 顺序）
                    mean = np.array([107.780, 136.565, 146.372])
                    std = np.array([70.402, 74.745, 75.099])

                # 取第一个 batch 的第一个视角的图像 (C, H, W)
                sample_img = x[0].cpu()  # 此时 x 形状为 (B*N, C, H, W)，取第一个

                # 反归一化：img_orig = img * std + mean
                sample_img = sample_img * torch.from_numpy(std).view(3,1,1) + torch.from_numpy(mean).view(3,1,1)
                sample_img = sample_img.clamp(0, 255).to(torch.uint8)  # 转为 0-255 并 uint8
                img_np = sample_img.permute(1,2,0).numpy()  # (H, W, C)

                # 保存图片
                import os
                from PIL import Image
                save_dir = "./debug_imgs"
                os.makedirs(save_dir, exist_ok=True)
                # 可添加迭代计数器，需要自行维护 self.iter_count（见下方说明）
                iter_count = getattr(self, 'iter_count', 0)
                save_path = os.path.join(save_dir, f"input_img_iter{iter_count}.png")
                Image.fromarray(img_np).save(save_path)
                print(f"[DEBUG] Saved input image to {save_path}")
                print(f"[DEBUG] Image shape: {img_np.shape}, range: [{img_np.min()}, {img_np.max()}]")

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        # ========== 调试：打印特征统计并阻塞 ==========
        if debug_img_feat:
            self.iter_count = 0
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print("\n[DEBUG] Swin Transformer + Neck output features:")
                # 注意 x 此时可能是多尺度输出（list/tuple）
                if isinstance(x, (list, tuple)):
                    for i, feat in enumerate(x):
                        print(f"  Scale {i}: shape {feat.shape}, "
                            f"mean {feat.mean().item():.6f}, std {feat.std().item():.6f}, "
                            f"min {feat.min().item():.6f}, max {feat.max().item():.6f}")
                        if torch.isnan(feat).any():
                            print(f"  !!! NaN detected at scale {i} !!!")
                        if torch.isinf(feat).any():
                            print(f"  !!! Inf detected at scale {i} !!!")
                else:
                    print(f"  Single tensor: shape {x.shape}, "
                        f"mean {x.mean().item():.6f}, std {x.std().item():.6f}, "
                        f"min {x.min().item():.6f}, max {x.max().item():.6f}")
                    
                feat_map = x[0].cpu()  # (C, H, W)
                C, H, W = feat_map.shape
                # 方法1：保存均值特征图
                mean_map = feat_map.mean(dim=0)  # (H, W)
                # 归一化到 [0,255] 方便保存
                mean_map_norm = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min() + 1e-8)
                mean_map_norm = (mean_map_norm * 255).byte().numpy()
                from PIL import Image
                import matplotlib.pyplot as plt
                Image.fromarray(mean_map_norm).save(f"./debug_imgs/feat_mean_iter{self.iter_count}.png")

                # 方法2：保存响应最大的通道（方差最大）
                channel_var = feat_map.var(dim=(1,2))  # 每个通道的方差
                max_var_ch = channel_var.argmax().item()
                single_ch = feat_map[max_var_ch]  # (H, W)
                single_ch_norm = (single_ch - single_ch.min()) / (single_ch.max() - single_ch.min() + 1e-8)
                single_ch_norm = (single_ch_norm * 255).byte().numpy()
                Image.fromarray(single_ch_norm).save(f"./debug_imgs/feat_maxvar_ch{max_var_ch}_iter{self.iter_count}.png")

                # 方法3：保存前8个通道拼成一张大图（便于快速浏览）
                ncols = 4
                nrows = 2
                fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6))
                for i in range(min(8, C)):
                    ax = axes[i//ncols, i%ncols]
                    ch_map = feat_map[i].detach().numpy()
                    ch_map_norm = (ch_map - ch_map.min()) / (ch_map.max() - ch_map.min() + 1e-8)
                    ax.imshow(ch_map_norm, cmap='viridis')
                    ax.set_title(f'ch{i}')
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(f"./debug_imgs/feat_first8ch_iter{self.iter_count}.png")
                plt.close()

                print(f"[DEBUG] Saved feature maps to ./debug_imgs/")
                # 阻塞，等待人工检查
                input(">>> Press Enter to continue training (after checking image and features) <<<")
                # 可选：递增迭代计数器，以便下次保存不同文件名
                self.iter_count = getattr(self, 'iter_count', 0) + 1

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        # 如果视图变换模块存在，则调用；否则直接返回图像特征
        if self.view_transform is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                x = self.view_transform(
                    x,
                    points,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    img_metas,
                )
        # 若 view_transform 为 None，则 x 保持 (B, N, C, H, W) 形状
        return x

    # def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
    #          batch_data_samples: List[Det3DDataSample],
    #          **kwargs) -> List[Det3DDataSample]:
    #     batch_input_metas = [item.metainfo for item in batch_data_samples]
    #     feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

    #     losses = dict()
    #     if self.with_bbox_head:
    #         bbox_loss = self.bbox_head.loss(feats, batch_data_samples, **kwargs)

    #     losses.update(bbox_loss)

    #     return losses
    
    # def extract_feat(self, batch_inputs_dict, batch_input_metas, **kwargs):
        # imgs = batch_inputs_dict.get('imgs', None)
        # points = batch_inputs_dict.get('points', None)
        # features = []
        
        # # 提取图像特征（如果存在）
        # if imgs is not None:
        #     imgs = imgs.contiguous()
        #     lidar2image, camera_intrinsics, camera2lidar = [], [], []
        #     img_aug_matrix, lidar_aug_matrix = [], []
        #     for i, meta in enumerate(batch_input_metas):
        #         lidar2image.append(meta['lidar2img'])
        #         camera_intrinsics.append(meta['cam2img'])
        #         camera2lidar.append(meta['cam2lidar'])
        #         img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
        #         lidar_aug_matrix.append(meta.get('lidar_aug_matrix', np.eye(4)))
            
        #     lidar2image = imgs.new_tensor(np.asarray(lidar2image))
        #     camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
        #     camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
        #     img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
        #     lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
        #     img_feature = self.extract_img_feat(imgs, deepcopy(points),
        #                                         lidar2image, camera_intrinsics,
        #                                         camera2lidar, img_aug_matrix,
        #                                         lidar_aug_matrix,
        #                                         batch_input_metas)
        #     features.append(img_feature)
        # # else:
        # #     # 图像不可用时，记录警告或直接跳过
        # #     print("Warning: No images in batch_inputs_dict, image branch will be ignored.")
        
        # # 提取点云特征（必须存在）
        # pts_feature = self.extract_pts_feat(batch_inputs_dict=batch_inputs_dict)
        # if pts_feature is None:
        #     raise RuntimeError("Point cloud feature extraction returned None")
        # features.append(pts_feature)
        
        # # 融合前检查
        # if len(features) == 1:
        #     # 只有点云特征
        #     x = features[0]
        # else:
        #     # 有图像和点云特征
        #     img_feat = features[0]
        #     pts_feat = features[1]
        #     # 统一空间尺寸
        #     # if img_feat.shape[2:] != pts_feat.shape[2:]:
        #     #     pts_feat = F.interpolate(pts_feat, size=img_feat.shape[2:],
        #     #                             mode='bilinear', align_corners=False)
        #     #     features[1] = pts_feat
        #     # print(f"Original img size: {img_feat.shape[2:]}")  # 输出原图像特征尺寸
        #     if img_feat.shape[2:] != pts_feat.shape[2:]:
        #         img_feat = F.interpolate(img_feat, size=pts_feat.shape[2:],
        #                                 mode='bilinear', align_corners=False)
        #         features[0] = img_feat
        #     if self.fusion_layer is not None:
        #         x = self.fusion_layer(features)
        #     else:
        #         x = torch.cat(features, dim=1)
        
        # # 经过 backbone 和 neck（如果有）
        # if self.backbone is not None:
        #     x = self.backbone(x)
        # if self.neck is not None:
        #     x = self.neck(x)
        # return x

    def extract_feat(self, batch_inputs_dict, batch_input_metas, **kwargs):
        if self.view_transform is None: # Bevpillars branch
            imgs = batch_inputs_dict.get('imgs', None)
            points = batch_inputs_dict.get('points', None)   # 原始点云列表
            img_feat = None
            pts_feat = None

            # 1. 提取图像特征（如果存在）
            if imgs is not None:
                imgs = imgs.contiguous()
                lidar2image, camera_intrinsics, camera2lidar = [], [], []
                img_aug_matrix, lidar_aug_matrix = [], []
                for i, meta in enumerate(batch_input_metas):
                    lidar2image.append(meta['lidar2img'])
                    camera_intrinsics.append(meta['cam2img'])
                    camera2lidar.append(meta['cam2lidar'])
                    img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                    lidar_aug_matrix.append(meta.get('lidar_aug_matrix', np.eye(4)))

                lidar2image = imgs.new_tensor(np.asarray(lidar2image))
                camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
                camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
                img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
                lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))

                img_feat = self.extract_img_feat(
                    imgs,
                    deepcopy(points),          # 避免修改原始数据
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    batch_input_metas,
                )

                img_feat = img_feat[:, 0, ...]

            # 2. 提取点云特征（必须存在）
            pts_feat = self.extract_pts_feat(batch_inputs_dict=batch_inputs_dict)
            if pts_feat is None:
                raise RuntimeError("Point cloud feature extraction returned None")

            if self.mode == 'mid': # 中期融合 视觉也输入backbone
                # 3. 融合图像和点云特征
                if self.fusion_layer is not None:
                    # 注意参数顺序：融合模块期望 (lidar_bev_feat, points, img_feat, img_metas)
                    # 如果图像特征不存在，则传入 None
                    x = self.fusion_layer(pts_feat, points, img_feat, batch_input_metas)
                else:
                    # 无融合层时，简单处理
                    if img_feat is not None:
                        # 需要将图像特征空间尺寸与点云 BEV 对齐
                        if img_feat.shape[2:] != pts_feat.shape[2:]:
                            img_feat = F.interpolate(
                                img_feat, size=pts_feat.shape[2:],
                                mode='bilinear', align_corners=False
                            )
                        x = torch.cat([img_feat, pts_feat], dim=1)
                    else:
                        x = pts_feat

                # 4. 通过 backbone 和 neck
                if self.backbone is not None:
                    x = self.backbone(x)
                if self.neck is not None:
                    x = self.neck(x)
                    
            elif self.mode == 'late': # 后期融合 视觉不输入backbone，直接输出bbox_head
                 # 3. 通过 backbone 和 neck
                if self.backbone is not None:
                    pts_feat = self.backbone(pts_feat)
                if self.neck is not None:
                    pts_feat = self.neck(pts_feat)
                # 3. 融合图像和点云特征
                if self.fusion_layer is not None:
                    # 注意参数顺序：融合模块期望 (lidar_bev_feat, points, img_feat, img_metas)
                    # 如果图像特征不存在，则传入 None
                    pts_feat = pts_feat[0]
                    x = self.fusion_layer(pts_feat, points, img_feat, batch_input_metas)
                    x = [x]
                else:
                    # 无融合层时，简单处理
                    if img_feat is not None:
                        # 需要将图像特征空间尺寸与点云 BEV 对齐
                        if img_feat.shape[2:] != pts_feat.shape[2:]:
                            img_feat = F.interpolate(
                                img_feat, size=pts_feat.shape[2:],
                                mode='bilinear', align_corners=False
                            )
                        x = torch.cat([img_feat, pts_feat], dim=1)
                    else:
                        x = pts_feat

            return x

        else: # 原始 BEVFusion 分支
            imgs = batch_inputs_dict.get('imgs', None)
            points = batch_inputs_dict.get('points', None)
            features = []
            
            # 提取图像特征（如果存在）
            if imgs is not None:
                imgs = imgs.contiguous()
                lidar2image, camera_intrinsics, camera2lidar = [], [], []
                img_aug_matrix, lidar_aug_matrix = [], []
                for i, meta in enumerate(batch_input_metas):
                    lidar2image.append(meta['lidar2img'])
                    camera_intrinsics.append(meta['cam2img'])
                    camera2lidar.append(meta['cam2lidar'])
                    img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                    lidar_aug_matrix.append(meta.get('lidar_aug_matrix', np.eye(4)))
                
                lidar2image = imgs.new_tensor(np.asarray(lidar2image))
                camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
                camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
                img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
                lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
                img_feature = self.extract_img_feat(imgs, deepcopy(points),
                                                    lidar2image, camera_intrinsics,
                                                    camera2lidar, img_aug_matrix,
                                                    lidar_aug_matrix,
                                                    batch_input_metas)
                features.append(img_feature)
            # else:
            #     # 图像不可用时，记录警告或直接跳过
            #     print("Warning: No images in batch_inputs_dict, image branch will be ignored.")
            
            # 提取点云特征（必须存在）
            pts_feature = self.extract_pts_feat(batch_inputs_dict=batch_inputs_dict)
            if pts_feature is None:
                raise RuntimeError("Point cloud feature extraction returned None")
            features.append(pts_feature)
            
            # 融合前检查
            if len(features) == 1:
                # 只有点云特征
                x = features[0]
            else:
                # 有图像和点云特征
                img_feat = features[0]
                pts_feat = features[1]
                # 统一空间尺寸
                # if img_feat.shape[2:] != pts_feat.shape[2:]:
                #     pts_feat = F.interpolate(pts_feat, size=img_feat.shape[2:],
                #                             mode='bilinear', align_corners=False)
                #     features[1] = pts_feat
                # print(f"Original img size: {img_feat.shape[2:]}")  # 输出原图像特征尺寸
                if img_feat.shape[2:] != pts_feat.shape[2:]:
                    img_feat = F.interpolate(img_feat, size=pts_feat.shape[2:],
                                            mode='bilinear', align_corners=False)
                    features[0] = img_feat
                if self.fusion_layer is not None:
                    x = self.fusion_layer(features)
                else:
                    x = torch.cat(features, dim=1)
            
            # 经过 backbone 和 neck（如果有）
            if self.backbone is not None:
                x = self.backbone(x)
            if self.neck is not None:
                x = self.neck(x)
            return x

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        losses = dict()
        # if self.with_bbox_head:
        bbox_loss = self.bbox_head.loss(feats, batch_data_samples,  **kwargs)

        losses.update(bbox_loss)

        return losses

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        # if self.with_bbox_head:
        outputs = self.bbox_head.predict(feats, batch_data_samples, **kwargs)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def _forward(self,
                 batch_inputs_dict: dict,
                 data_samples: OptSampleList = None,
                 **kwargs) -> Tuple[List[torch.Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        batch_input_metas = [item.metainfo for item in data_samples]
        x = self.extract_feat(batch_inputs_dict, batch_input_metas)
        results = self.bbox_head.forward(x)
        return results