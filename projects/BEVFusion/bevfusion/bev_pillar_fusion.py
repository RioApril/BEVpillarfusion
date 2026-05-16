# import torch
# import torch.nn as nn
# from mmdet3d.registry import MODELS
# import torch.nn.functional as F
# # from mmdet3d.structures import points_cam2img
# import numpy as np

# @MODELS.register_module()
# class BEVPillarFusion(nn.Module):
#     def __init__(self,
#                  image_channels,          # 原始图像特征通道数，例如256
#                  lidar_bev_channels,      # 点云BEV特征通道数
#                  out_channels,            # 融合输出通道数
#                  bev_h, bev_w,
#                  point_cloud_range,
#                  voxel_size,
#                  num_samples=20,
#                  attn_hidden_dim=64,
#                  img_downsample_channels=32,  # 聚合后降维到的通道数
#                  **kwargs):
#         super().__init__()
#         self.image_channels = image_channels
#         self.lidar_bev_channels = lidar_bev_channels
#         self.out_channels = out_channels
#         self.bev_h = bev_h
#         self.bev_w = bev_w
#         self.point_cloud_range = point_cloud_range
#         self.voxel_size = voxel_size
#         self.num_samples = num_samples
#         self.attn_hidden_dim = attn_hidden_dim
#         self.img_downsample_channels = img_downsample_channels

#         # 网格物理参数
#         self.x_min, self.y_min, self.z_min = point_cloud_range[:3]
#         self.x_max, self.y_max, self.z_max = point_cloud_range[3:]
#         self.vx, self.vy, self.vz = voxel_size
#         self.nx = int((self.x_max - self.x_min) / self.vx)
#         self.ny = int((self.y_max - self.y_min) / self.vy)
#         self.bev_h = bev_h if bev_h is not None else self.ny
#         self.bev_w = bev_w if bev_w is not None else self.nx

#         # 预计算网格中心坐标
#         x_centers = torch.arange(self.bev_w, dtype=torch.float32) * self.vx + self.x_min + self.vx/2
#         y_centers = torch.arange(self.bev_h, dtype=torch.float32) * self.vy + self.y_min + self.vy/2
#         self.register_buffer('x_centers', x_centers)
#         self.register_buffer('y_centers', y_centers)

#         # 注意力聚合模块（输入为原始高维特征）
#         self.attention = nn.Sequential(
#             nn.Linear(image_channels, attn_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(attn_hidden_dim, 1)
#         )

#         # 聚合后的降维层（高维 -> 低维）
#         self.agg_downsample = nn.Linear(image_channels, img_downsample_channels)

#         # 图像特征投影（将降维后的图像BEV特征投影到点云特征维度）
#         self.img_proj = nn.Linear(img_downsample_channels, lidar_bev_channels) if out_channels == lidar_bev_channels else None

#         # 融合卷积
#         if self.img_proj is not None:
#             concat_channels = lidar_bev_channels + lidar_bev_channels
#         else:
#             concat_channels = lidar_bev_channels + img_downsample_channels
#         self.fusion_conv = nn.Conv2d(concat_channels, out_channels, 1)

#     def forward(self, lidar_bev_feat, points, img_feat, img_metas):
#         B, C2, H, W = lidar_bev_feat.shape
#         device = lidar_bev_feat.device

#         # 处理图像特征维度（可能为5维，取第一个相机）
#         if img_feat.dim() == 5:
#             img_feat = img_feat[:, 0, ...]          # (B, C_orig, H_img, W_img)
#         # 注意：不再进行全局降维，保留原始高维特征图
#         C_orig, H_img, W_img = img_feat.shape[1], img_feat.shape[2], img_feat.shape[3]

#         # 初始化图像BEV特征图（通道数为聚合后降维的通道数）
#         img_bev_feat = torch.zeros(B, self.img_downsample_channels, H, W, device=device)

#         # 预计算网格中心矩阵
#         x_center_map = self.x_centers.view(1, -1).expand(H, -1)   # (H, W)
#         y_center_map = self.y_centers.view(-1, 1).expand(-1, W)   # (H, W)

#         for batch_idx in range(B):
#             pts = points[batch_idx][:, :3]                      # (M, 3)
#             if pts.shape[0] == 0:
#                 continue

#             # 获取相机投影矩阵（第一个相机）
#             lidar2img = img_metas[batch_idx]['lidar2img'][0]
#             lidar2img = torch.tensor(lidar2img, dtype=torch.float32, device=device)

#             # ---------- 1. 真实点分配与投影 ----------
#             grid_x = ((pts[:, 0] - self.x_min) / self.vx).long()
#             grid_y = ((pts[:, 1] - self.y_min) / self.vy).long()
#             valid = (grid_x >= 0) & (grid_x < W) & (grid_y >= 0) & (grid_y < H)
#             grid_x = grid_x[valid]
#             grid_y = grid_y[valid]
#             pts_valid = pts[valid]

#             if pts_valid.shape[0] == 0:
#                 continue

#             # 批量投影
#             ones = torch.ones(pts_valid.shape[0], 1, device=device)
#             pts_homo = torch.cat([pts_valid, ones], dim=1)
#             img_pts = pts_homo @ lidar2img.T
#             u = img_pts[:, 0] / img_pts[:, 2]
#             v = img_pts[:, 1] / img_pts[:, 2]
#             depth = img_pts[:, 2]

#             valid_proj = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img) & (depth > 0)
#             u = u[valid_proj]
#             v = v[valid_proj]
#             grid_x = grid_x[valid_proj]
#             grid_y = grid_y[valid_proj]
#             pts_valid = pts_valid[valid_proj]
#             if pts_valid.shape[0] == 0:
#                 continue

#             # 批量双线性采样（得到高维特征）
#             img_feat_b = img_feat[batch_idx]                     # (C_orig, H_img, W_img)
#             sampled_feat = self._batch_bilinear_sample(img_feat_b, u, v)  # (K, C_orig)

#             # 按网格分组
#             grid_indices = grid_y * W + grid_x
#             unique_grids, inverse, counts = torch.unique(grid_indices, return_inverse=True, return_counts=True)

#             selected_per_grid = {}
#             for idx, gid in enumerate(unique_grids):
#                 mask = inverse == idx
#                 pts_in_grid = pts_valid[mask]
#                 feats_in_grid = sampled_feat[mask]
#                 num_pts = pts_in_grid.shape[0]
#                 if num_pts >= self.num_samples:
#                     chosen = np.random.choice(num_pts, self.num_samples, replace=False)
#                     selected_per_grid[gid.item()] = (pts_in_grid[chosen], feats_in_grid[chosen])
#                 else:
#                     selected_per_grid[gid.item()] = (pts_in_grid, feats_in_grid)

#             # ---------- 2. 虚拟点生成与投影（针对需要补齐的网格）----------
#             need_virtual = set()
#             for gid, (pts_g, _) in selected_per_grid.items():
#                 if pts_g.shape[0] < self.num_samples:
#                     need_virtual.add(gid)

#             virtual_points_dict = {}
#             if need_virtual:
#                 gx_list = [gid % W for gid in need_virtual]
#                 gy_list = [gid // W for gid in need_virtual]
#                 x_centers_needed = x_center_map[gy_list, gx_list]
#                 y_centers_needed = y_center_map[gy_list, gx_list]

#                 K_max = self.num_samples
#                 ratios = torch.linspace(1/(K_max+1), K_max/(K_max+1), K_max, device=device)
#                 z_virt = self.z_min + ratios * (self.z_max - self.z_min)
#                 x_expand = x_centers_needed[:, None].expand(-1, K_max)
#                 y_expand = y_centers_needed[:, None].expand(-1, K_max)
#                 z_expand = z_virt[None, :].expand(len(need_virtual), -1)
#                 virtual_xyz = torch.stack([x_expand, y_expand, z_expand], dim=-1)  # (N_g, K_max, 3)

#                 N_g = len(need_virtual)
#                 virtual_flat = virtual_xyz.reshape(-1, 3)
#                 ones_v = torch.ones(virtual_flat.shape[0], 1, device=device)
#                 v_homo = torch.cat([virtual_flat, ones_v], dim=1)
#                 v_img = v_homo @ lidar2img.T
#                 v_u = v_img[:, 0] / v_img[:, 2]
#                 v_v = v_img[:, 1] / v_img[:, 2]
#                 v_depth = v_img[:, 2]

#                 v_valid = (v_u >= 0) & (v_u < W_img) & (v_v >= 0) & (v_v < H_img) & (v_depth > 0)
#                 v_feat = self._batch_bilinear_sample(img_feat_b, v_u, v_v)  # (N_g*K_max, C_orig)
#                 v_feat[~v_valid] = 0.0
#                 v_feat = v_feat.reshape(N_g, K_max, C_orig)

#                 for i, gid in enumerate(need_virtual):
#                     real_cnt = selected_per_grid[gid][0].shape[0]
#                     need_cnt = self.num_samples - real_cnt
#                     if need_cnt > 0:
#                         virtual_points_dict[gid] = v_feat[i, :need_cnt, :]
#                     else:
#                         virtual_points_dict[gid] = torch.empty(0, C_orig, device=device)

#             # ---------- 3. 每个网格：聚合高维特征 -> 降维 -> 填入BEV图 ----------
#             for gy in range(H):
#                 for gx in range(W):
#                     gid = gy * W + gx
#                     real_pts, real_feat = selected_per_grid.get(gid, (torch.empty(0,3,device=device), torch.empty(0,C_orig,device=device)))
#                     real_cnt = real_feat.shape[0]
#                     if real_cnt == 0 and gid not in virtual_points_dict:
#                         continue

#                     all_feats = [real_feat]
#                     if gid in virtual_points_dict:
#                         virt_feat = virtual_points_dict[gid]
#                         if virt_feat.shape[0] > 0:
#                             all_feats.append(virt_feat)
#                     all_feats = torch.cat(all_feats, dim=0)          # (total, C_orig)
#                     total = all_feats.shape[0]
#                     if total < self.num_samples:
#                         pad = torch.zeros(self.num_samples - total, C_orig, device=device)
#                         all_feats = torch.cat([all_feats, pad], dim=0)
#                     elif total > self.num_samples:
#                         idx = torch.randperm(total)[:self.num_samples]
#                         all_feats = all_feats[idx]

#                     # 注意力聚合（在高维空间）
#                     attn_weights = self.attention(all_feats)        # (N, 1)
#                     attn_weights = torch.softmax(attn_weights, dim=0)
#                     agg_feat_high = (all_feats * attn_weights).sum(dim=0)  # (C_orig,)

#                     # 降维到低维
#                     agg_feat_low = self.agg_downsample(agg_feat_high)     # (img_downsample_channels,)

#                     img_bev_feat[batch_idx, :, gy, gx] = agg_feat_low

#         # ---------- 4. 投影与融合 ----------
#         if self.img_proj is not None:
#             img_bev_feat = img_bev_feat.permute(0, 2, 3, 1)   # (B, H, W, C_low)
#             img_bev_feat = self.img_proj(img_bev_feat)        # (B, H, W, C2)
#             img_bev_feat = img_bev_feat.permute(0, 3, 1, 2)   # (B, C2, H, W)

#         fusion_feat = torch.cat([lidar_bev_feat, img_bev_feat], dim=1)
#         fusion_feat = self.fusion_conv(fusion_feat)
#         return fusion_feat

#     def _batch_bilinear_sample(self, feat_map, u, v):
#         N = u.shape[0]
#         if N == 0:
#             return torch.empty(0, feat_map.shape[0], device=feat_map.device)
#         C, H, W = feat_map.shape
#         u_norm = 2.0 * u / (W - 1) - 1.0
#         v_norm = 2.0 * v / (H - 1) - 1.0
#         grid = torch.stack([u_norm, v_norm], dim=1).view(1, N, 1, 2)
#         feat_batch = feat_map.unsqueeze(0)
#         sampled = F.grid_sample(feat_batch, grid, mode='bilinear', align_corners=False)
#         sampled = sampled.squeeze(0).squeeze(2)
#         return sampled.permute(1, 0)

# import torch
# import torch.nn as nn
# from mmdet3d.registry import MODELS
# import torch.nn.functional as F
# import numpy as np

# @MODELS.register_module()
# class BEVPillarFusion(nn.Module):
#     def __init__(self,
#                  image_channels,          # 原始图像特征通道数，例如256
#                  lidar_bev_channels,      # 点云BEV特征通道数
#                  out_channels,            # 融合输出通道数
#                  bev_h, bev_w,
#                  point_cloud_range,
#                  voxel_size,
#                  num_samples=20,
#                  attn_hidden_dim=64,
#                  img_downsample_channels=32,  # 聚合后降维到的通道数
#                  **kwargs):
#         super().__init__()
#         self.image_channels = image_channels
#         self.lidar_bev_channels = lidar_bev_channels
#         self.out_channels = out_channels
#         self.bev_h = bev_h
#         self.bev_w = bev_w
#         self.point_cloud_range = point_cloud_range
#         self.voxel_size = voxel_size
#         self.num_samples = num_samples
#         self.attn_hidden_dim = attn_hidden_dim
#         self.img_downsample_channels = img_downsample_channels

#         # 网格物理参数
#         self.x_min, self.y_min, self.z_min = point_cloud_range[:3]
#         self.x_max, self.y_max, self.z_max = point_cloud_range[3:]
#         self.vx, self.vy, self.vz = voxel_size
#         self.nx = int((self.x_max - self.x_min) / self.vx)
#         self.ny = int((self.y_max - self.y_min) / self.vy)
#         self.bev_h = bev_h if bev_h is not None else self.ny
#         self.bev_w = bev_w if bev_w is not None else self.nx

#         # 预计算网格中心坐标
#         x_centers = torch.arange(self.bev_w, dtype=torch.float32) * self.vx + self.x_min + self.vx/2
#         y_centers = torch.arange(self.bev_h, dtype=torch.float32) * self.vy + self.y_min + self.vy/2
#         self.register_buffer('x_centers', x_centers)
#         self.register_buffer('y_centers', y_centers)

#         # 注意力聚合模块（输入为原始高维特征）
#         self.attention = nn.Sequential(
#             nn.Linear(image_channels, attn_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(attn_hidden_dim, 1)
#         )

#         # 聚合后的降维层（高维 -> 低维）
#         self.agg_downsample = nn.Linear(image_channels, img_downsample_channels)

#         # 图像特征投影（将降维后的图像BEV特征投影到点云特征维度）
#         self.img_proj = nn.Linear(img_downsample_channels, lidar_bev_channels) if out_channels == lidar_bev_channels else None

#         # 融合卷积
#         if self.img_proj is not None:
#             concat_channels = lidar_bev_channels + lidar_bev_channels
#         else:
#             concat_channels = lidar_bev_channels + img_downsample_channels
#         self.fusion_conv = nn.Conv2d(concat_channels, out_channels, 1)

#     def forward(self, lidar_bev_feat, points, img_feat, img_metas):
#         B, C2, H, W = lidar_bev_feat.shape
#         device = lidar_bev_feat.device

#         # 处理图像特征维度（可能为5维，取第一个相机）
#         if img_feat.dim() == 5:
#             img_feat = img_feat[:, 0, ...]          # (B, C_orig, H_img, W_img)
#         C_orig, H_img, W_img = img_feat.shape[1], img_feat.shape[2], img_feat.shape[3]

#         # 初始化图像BEV特征图（通道数为聚合后降维的通道数）
#         img_bev_feat = torch.zeros(B, self.img_downsample_channels, H, W, device=device)

#         # 预计算网格中心矩阵
#         x_center_map = self.x_centers.view(1, -1).expand(H, -1)   # (H, W)
#         y_center_map = self.y_centers.view(-1, 1).expand(-1, W)   # (H, W)

#         for batch_idx in range(B):
#             pts = points[batch_idx][:, :3]                      # (M, 3)
#             if pts.shape[0] == 0:
#                 continue

#             # ========== 关键修改：合成总投影矩阵（支持数据增强） ==========
#             # 1. 获取原始投影矩阵（LiDAR → 原始图像像素）
#             lidar2img = np.array(img_metas[batch_idx]['lidar2img'][0])   # 4x4
#             lidar2img = lidar2img.reshape(4, 4)

#             # 2. 获取增强矩阵
#             lidar_aug = np.eye(4)
#             if 'lidar_aug_matrix' in img_metas[batch_idx]:
#                 lidar_aug = np.array(img_metas[batch_idx]['lidar_aug_matrix'])
#             img_aug = np.eye(4)
#             if 'img_aug_matrix' in img_metas[batch_idx]:
#                 img_aug = np.array(img_metas[batch_idx]['img_aug_matrix'][0])  # 第一个相机

#             # 3. 计算总投影：p_img_aug = img_aug @ lidar2img @ inv(lidar_aug) @ P_lidar_aug
#             lidar_aug_inv = np.linalg.inv(lidar_aug)
#             total_proj_np = img_aug @ lidar2img @ lidar_aug_inv   # 4x4
#             total_proj = torch.tensor(total_proj_np, dtype=torch.float32, device=device)

#             # ---------- 1. 真实点分配与投影 ----------
#             grid_x = ((pts[:, 0] - self.x_min) / self.vx).long()
#             grid_y = ((pts[:, 1] - self.y_min) / self.vy).long()
#             valid = (grid_x >= 0) & (grid_x < W) & (grid_y >= 0) & (grid_y < H)
#             grid_x = grid_x[valid]
#             grid_y = grid_y[valid]
#             pts_valid = pts[valid]

#             if pts_valid.shape[0] == 0:
#                 continue

#             # 使用总投影矩阵进行批量投影
#             ones = torch.ones(pts_valid.shape[0], 1, device=device)
#             pts_homo = torch.cat([pts_valid, ones], dim=1)
#             img_pts = pts_homo @ total_proj.T
#             u = img_pts[:, 0] / img_pts[:, 2]
#             v = img_pts[:, 1] / img_pts[:, 2]
#             depth = img_pts[:, 2]

#             valid_proj = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img) & (depth > 0)
#             u = u[valid_proj]
#             v = v[valid_proj]
#             grid_x = grid_x[valid_proj]
#             grid_y = grid_y[valid_proj]
#             pts_valid = pts_valid[valid_proj]
#             if pts_valid.shape[0] == 0:
#                 continue

#             # 批量双线性采样（得到高维特征）
#             img_feat_b = img_feat[batch_idx]                     # (C_orig, H_img, W_img)
#             sampled_feat = self._batch_bilinear_sample(img_feat_b, u, v)  # (K, C_orig)

#             # 按网格分组
#             grid_indices = grid_y * W + grid_x
#             unique_grids, inverse, counts = torch.unique(grid_indices, return_inverse=True, return_counts=True)

#             selected_per_grid = {}
#             for idx, gid in enumerate(unique_grids):
#                 mask = inverse == idx
#                 pts_in_grid = pts_valid[mask]
#                 feats_in_grid = sampled_feat[mask]
#                 num_pts = pts_in_grid.shape[0]
#                 if num_pts >= self.num_samples:
#                     chosen = np.random.choice(num_pts, self.num_samples, replace=False)
#                     selected_per_grid[gid.item()] = (pts_in_grid[chosen], feats_in_grid[chosen])
#                 else:
#                     selected_per_grid[gid.item()] = (pts_in_grid, feats_in_grid)

#             # ---------- 2. 虚拟点生成与投影（针对需要补齐的网格）----------
#             need_virtual = set()
#             for gid, (pts_g, _) in selected_per_grid.items():
#                 if pts_g.shape[0] < self.num_samples:
#                     need_virtual.add(gid)

#             virtual_points_dict = {}
#             if need_virtual:
#                 gx_list = [gid % W for gid in need_virtual]
#                 gy_list = [gid // W for gid in need_virtual]
#                 x_centers_needed = x_center_map[gy_list, gx_list]
#                 y_centers_needed = y_center_map[gy_list, gx_list]

#                 K_max = self.num_samples
#                 ratios = torch.linspace(1/(K_max+1), K_max/(K_max+1), K_max, device=device)
#                 z_virt = self.z_min + ratios * (self.z_max - self.z_min)
#                 x_expand = x_centers_needed[:, None].expand(-1, K_max)
#                 y_expand = y_centers_needed[:, None].expand(-1, K_max)
#                 z_expand = z_virt[None, :].expand(len(need_virtual), -1)
#                 virtual_xyz = torch.stack([x_expand, y_expand, z_expand], dim=-1)  # (N_g, K_max, 3)

#                 N_g = len(need_virtual)
#                 virtual_flat = virtual_xyz.reshape(-1, 3)
#                 ones_v = torch.ones(virtual_flat.shape[0], 1, device=device)
#                 v_homo = torch.cat([virtual_flat, ones_v], dim=1)
#                 # 使用同一个总投影矩阵
#                 v_img = v_homo @ total_proj.T
#                 v_u = v_img[:, 0] / v_img[:, 2]
#                 v_v = v_img[:, 1] / v_img[:, 2]
#                 v_depth = v_img[:, 2]

#                 v_valid = (v_u >= 0) & (v_u < W_img) & (v_v >= 0) & (v_v < H_img) & (v_depth > 0)
#                 v_feat = self._batch_bilinear_sample(img_feat_b, v_u, v_v)  # (N_g*K_max, C_orig)
#                 v_feat[~v_valid] = 0.0
#                 v_feat = v_feat.reshape(N_g, K_max, C_orig)

#                 for i, gid in enumerate(need_virtual):
#                     real_cnt = selected_per_grid[gid][0].shape[0]
#                     need_cnt = self.num_samples - real_cnt
#                     if need_cnt > 0:
#                         virtual_points_dict[gid] = v_feat[i, :need_cnt, :]
#                     else:
#                         virtual_points_dict[gid] = torch.empty(0, C_orig, device=device)

#             # ---------- 3. 每个网格：聚合高维特征 -> 降维 -> 填入BEV图 ----------
#             for gy in range(H):
#                 for gx in range(W):
#                     gid = gy * W + gx
#                     real_pts, real_feat = selected_per_grid.get(gid, (torch.empty(0,3,device=device), torch.empty(0,C_orig,device=device)))
#                     real_cnt = real_feat.shape[0]
#                     if real_cnt == 0 and gid not in virtual_points_dict:
#                         continue

#                     all_feats = [real_feat]
#                     if gid in virtual_points_dict:
#                         virt_feat = virtual_points_dict[gid]
#                         if virt_feat.shape[0] > 0:
#                             all_feats.append(virt_feat)
#                     all_feats = torch.cat(all_feats, dim=0)          # (total, C_orig)
#                     total = all_feats.shape[0]
#                     if total < self.num_samples:
#                         pad = torch.zeros(self.num_samples - total, C_orig, device=device)
#                         all_feats = torch.cat([all_feats, pad], dim=0)
#                     elif total > self.num_samples:
#                         idx = torch.randperm(total)[:self.num_samples]
#                         all_feats = all_feats[idx]

#                     # 注意力聚合（在高维空间）
#                     attn_weights = self.attention(all_feats)        # (N, 1)
#                     attn_weights = torch.softmax(attn_weights, dim=0)
#                     agg_feat_high = (all_feats * attn_weights).sum(dim=0)  # (C_orig,)

#                     # 降维到低维
#                     agg_feat_low = self.agg_downsample(agg_feat_high)     # (img_downsample_channels,)

#                     img_bev_feat[batch_idx, :, gy, gx] = agg_feat_low

#         # ---------- 4. 投影与融合 ----------
#         if self.img_proj is not None:
#             img_bev_feat = img_bev_feat.permute(0, 2, 3, 1)   # (B, H, W, C_low)
#             img_bev_feat = self.img_proj(img_bev_feat)        # (B, H, W, C2)
#             img_bev_feat = img_bev_feat.permute(0, 3, 1, 2)   # (B, C2, H, W)

#         fusion_feat = torch.cat([lidar_bev_feat, img_bev_feat], dim=1)
#         fusion_feat = self.fusion_conv(fusion_feat)
#         return fusion_feat

#     def _batch_bilinear_sample(self, feat_map, u, v):
#         N = u.shape[0]
#         if N == 0:
#             return torch.empty(0, feat_map.shape[0], device=feat_map.device)
#         C, H, W = feat_map.shape
#         u_norm = 2.0 * u / (W - 1) - 1.0
#         v_norm = 2.0 * v / (H - 1) - 1.0
#         grid = torch.stack([u_norm, v_norm], dim=1).view(1, N, 1, 2)
#         feat_batch = feat_map.unsqueeze(0)
#         sampled = F.grid_sample(feat_batch, grid, mode='bilinear', align_corners=False)
#         sampled = sampled.squeeze(0).squeeze(2)
#         return sampled.permute(1, 0)

# ============================
# 2026.4.14

# import torch
# import torch.nn as nn
# from mmdet3d.registry import MODELS
# import torch.nn.functional as F
# import numpy as np

# @MODELS.register_module()
# class BEVPillarFusion(nn.Module):
#     def __init__(self,
#                  image_channels,          # 原始图像特征通道数，例如256
#                  lidar_bev_channels,      # 点云BEV特征通道数
#                  out_channels,            # 融合输出通道数
#                  bev_h, bev_w,
#                  point_cloud_range,
#                  voxel_size,
#                  num_samples=20,
#                  attn_hidden_dim=64,
#                  img_downsample_channels=32,  # 聚合后降维到的通道数
#                  **kwargs):
#         super().__init__()
#         self.image_channels = image_channels
#         self.lidar_bev_channels = lidar_bev_channels
#         self.out_channels = out_channels
#         self.bev_h = bev_h
#         self.bev_w = bev_w
#         self.point_cloud_range = point_cloud_range
#         self.voxel_size = voxel_size
#         self.num_samples = num_samples
#         self.attn_hidden_dim = attn_hidden_dim
#         self.img_downsample_channels = img_downsample_channels

#         # 网格物理参数
#         self.x_min, self.y_min, self.z_min = point_cloud_range[:3]
#         self.x_max, self.y_max, self.z_max = point_cloud_range[3:]
#         self.vx, self.vy, self.vz = voxel_size
#         self.nx = int((self.x_max - self.x_min) / self.vx)
#         self.ny = int((self.y_max - self.y_min) / self.vy)
#         self.bev_h = bev_h if bev_h is not None else self.ny
#         self.bev_w = bev_w if bev_w is not None else self.nx

#         # 预计算网格中心坐标
#         x_centers = torch.arange(self.bev_w, dtype=torch.float32) * self.vx + self.x_min + self.vx/2
#         y_centers = torch.arange(self.bev_h, dtype=torch.float32) * self.vy + self.y_min + self.vy/2
#         self.register_buffer('x_centers', x_centers)
#         self.register_buffer('y_centers', y_centers)

#         # 注意力聚合模块（输入为原始高维特征）
#         self.attention = nn.Sequential(
#             nn.Linear(image_channels, attn_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(attn_hidden_dim, 1)
#         )

#         # 聚合后的降维层（高维 -> 低维）
#         self.agg_downsample = nn.Linear(image_channels, img_downsample_channels)

#         # 融合卷积：输入通道 = 点云BEV通道 + 图像下采样通道
#         concat_channels = lidar_bev_channels + img_downsample_channels
#         self.fusion_conv = nn.Conv2d(concat_channels, out_channels, 1)

#     def forward(self, lidar_bev_feat, points, img_feat, img_metas):
#         B, C2, H, W = lidar_bev_feat.shape
#         device = lidar_bev_feat.device

#         # 处理图像特征维度（可能为5维，取第一个相机）
#         if img_feat.dim() == 5:
#             img_feat = img_feat[:, 0, ...]          # (B, C_orig, H_img, W_img)
#         C_orig, H_img, W_img = img_feat.shape[1], img_feat.shape[2], img_feat.shape[3]

#         # 初始化图像BEV特征图（通道数为聚合后降维的通道数）
#         img_bev_feat = torch.zeros(B, self.img_downsample_channels, H, W, device=device)

#         # 预计算网格中心矩阵
#         x_center_map = self.x_centers.view(1, -1).expand(H, -1)   # (H, W)
#         y_center_map = self.y_centers.view(-1, 1).expand(-1, W)   # (H, W)

#         print("BEV Pillars Fusion in!")
#         for batch_idx in range(B):
#             pts = points[batch_idx][:, :3]                      # (M, 3)
            
#             if pts.shape[0] == 0:
#                 continue

#             # ========== 合成总投影矩阵（支持数据增强） ==========
#             # 1. 获取原始投影矩阵（LiDAR → 原始图像像素）
#             lidar2img = np.array(img_metas[batch_idx]['lidar2img'][0])   # 4x4
#             lidar2img = lidar2img.reshape(4, 4)

#             # 2. 获取增强矩阵
#             lidar_aug = np.eye(4)
#             if 'lidar_aug_matrix' in img_metas[batch_idx]:
#                 lidar_aug = np.array(img_metas[batch_idx]['lidar_aug_matrix'])
#             img_aug = np.eye(4)
#             if 'img_aug_matrix' in img_metas[batch_idx]:
#                 img_aug = np.array(img_metas[batch_idx]['img_aug_matrix'][0])  # 第一个相机

#             # 3. 计算总投影：p_img_aug = img_aug @ lidar2img @ inv(lidar_aug) @ P_lidar_aug
#             lidar_aug_inv = np.linalg.inv(lidar_aug)
#             total_proj_np = img_aug @ lidar2img @ lidar_aug_inv   # 4x4
#             total_proj = torch.tensor(total_proj_np, dtype=torch.float32, device=device)

#             # ---------- 1. 真实点分配与投影 ----------
#             grid_x = ((pts[:, 0] - self.x_min) / self.vx).long()
#             grid_y = ((pts[:, 1] - self.y_min) / self.vy).long()
#             valid = (grid_x >= 0) & (grid_x < W) & (grid_y >= 0) & (grid_y < H)
#             grid_x = grid_x[valid]
#             grid_y = grid_y[valid]
#             pts_valid = pts[valid]

#             if pts_valid.shape[0] == 0:
#                 continue

#             # 使用总投影矩阵进行批量投影
#             ones = torch.ones(pts_valid.shape[0], 1, device=device)
#             pts_homo = torch.cat([pts_valid, ones], dim=1)
#             img_pts = pts_homo @ total_proj.T
#             u = img_pts[:, 0] / img_pts[:, 2]
#             v = img_pts[:, 1] / img_pts[:, 2]
#             depth = img_pts[:, 2]

#             valid_proj = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img) & (depth > 0)
#             u = u[valid_proj]
#             v = v[valid_proj]
#             grid_x = grid_x[valid_proj]
#             grid_y = grid_y[valid_proj]
#             pts_valid = pts_valid[valid_proj]
#             if pts_valid.shape[0] == 0:
#                 continue

#             # 批量双线性采样（得到高维特征）
#             img_feat_b = img_feat[batch_idx]                     # (C_orig, H_img, W_img)
#             sampled_feat = self._batch_bilinear_sample(img_feat_b, u, v)  # (K, C_orig)

#             # 按网格分组
#             grid_indices = grid_y * W + grid_x
#             unique_grids, inverse, counts = torch.unique(grid_indices, return_inverse=True, return_counts=True)

#             selected_per_grid = {}
#             for idx, gid in enumerate(unique_grids):
#                 mask = inverse == idx
#                 pts_in_grid = pts_valid[mask]
#                 feats_in_grid = sampled_feat[mask]
#                 num_pts = pts_in_grid.shape[0]
#                 if num_pts >= self.num_samples:
#                     chosen = np.random.choice(num_pts, self.num_samples, replace=False)
#                     selected_per_grid[gid.item()] = (pts_in_grid[chosen], feats_in_grid[chosen])
#                 else:
#                     selected_per_grid[gid.item()] = (pts_in_grid, feats_in_grid)

#             # ---------- 2. 虚拟点生成与投影（针对需要补齐的网格）----------
#             need_virtual = set()
#             for gid, (pts_g, _) in selected_per_grid.items():
#                 if pts_g.shape[0] < self.num_samples:
#                     need_virtual.add(gid)

#             virtual_points_dict = {}
#             if need_virtual:
#                 gx_list = [gid % W for gid in need_virtual]
#                 gy_list = [gid // W for gid in need_virtual]
#                 x_centers_needed = x_center_map[gy_list, gx_list]
#                 y_centers_needed = y_center_map[gy_list, gx_list]

#                 K_max = self.num_samples
#                 ratios = torch.linspace(1/(K_max+1), K_max/(K_max+1), K_max, device=device)
#                 z_virt = self.z_min + ratios * (self.z_max - self.z_min)
#                 x_expand = x_centers_needed[:, None].expand(-1, K_max)
#                 y_expand = y_centers_needed[:, None].expand(-1, K_max)
#                 z_expand = z_virt[None, :].expand(len(need_virtual), -1)
#                 virtual_xyz = torch.stack([x_expand, y_expand, z_expand], dim=-1)  # (N_g, K_max, 3)

#                 N_g = len(need_virtual)
#                 virtual_flat = virtual_xyz.reshape(-1, 3)
#                 ones_v = torch.ones(virtual_flat.shape[0], 1, device=device)
#                 v_homo = torch.cat([virtual_flat, ones_v], dim=1)
#                 # 使用同一个总投影矩阵
#                 v_img = v_homo @ total_proj.T
#                 v_u = v_img[:, 0] / v_img[:, 2]
#                 v_v = v_img[:, 1] / v_img[:, 2]
#                 v_depth = v_img[:, 2]

#                 v_valid = (v_u >= 0) & (v_u < W_img) & (v_v >= 0) & (v_v < H_img) & (v_depth > 0)
#                 v_feat = self._batch_bilinear_sample(img_feat_b, v_u, v_v)  # (N_g*K_max, C_orig)
#                 v_feat[~v_valid] = 0.0
#                 v_feat = v_feat.reshape(N_g, K_max, C_orig)

#                 for i, gid in enumerate(need_virtual):
#                     real_cnt = selected_per_grid[gid][0].shape[0]
#                     need_cnt = self.num_samples - real_cnt
#                     if need_cnt > 0:
#                         virtual_points_dict[gid] = v_feat[i, :need_cnt, :]
#                     else:
#                         virtual_points_dict[gid] = torch.empty(0, C_orig, device=device)

#             # ---------- 3. 每个网格：聚合高维特征 -> 降维 -> 填入BEV图 ----------
#             for gy in range(H):
#                 for gx in range(W):
#                     gid = gy * W + gx
#                     real_pts, real_feat = selected_per_grid.get(gid, (torch.empty(0,3,device=device), torch.empty(0,C_orig,device=device)))
#                     real_cnt = real_feat.shape[0]
#                     if real_cnt == 0 and gid not in virtual_points_dict:
#                         continue

#                     all_feats = [real_feat]
#                     if gid in virtual_points_dict:
#                         virt_feat = virtual_points_dict[gid]
#                         if virt_feat.shape[0] > 0:
#                             all_feats.append(virt_feat)
#                     all_feats = torch.cat(all_feats, dim=0)          # (total, C_orig)
#                     total = all_feats.shape[0]
#                     if total < self.num_samples:
#                         pad = torch.zeros(self.num_samples - total, C_orig, device=device)
#                         all_feats = torch.cat([all_feats, pad], dim=0)
#                     elif total > self.num_samples:
#                         idx = torch.randperm(total)[:self.num_samples]
#                         all_feats = all_feats[idx]

#                     # 注意力聚合（在高维空间）
#                     attn_weights = self.attention(all_feats)        # (N, 1)
#                     attn_weights = torch.softmax(attn_weights, dim=0)
#                     agg_feat_high = (all_feats * attn_weights).sum(dim=0)  # (C_orig,)

#                     # 降维到低维
#                     agg_feat_low = self.agg_downsample(agg_feat_high)     # (img_downsample_channels,)

#                     img_bev_feat[batch_idx, :, gy, gx] = agg_feat_low

#         # ---------- 4. 融合：直接拼接点云BEV特征和图像BEV特征，再卷积输出 ----------
#         # img_bev_feat 形状: (B, img_downsample_channels, H, W)
#         # lidar_bev_feat 形状: (B, lidar_bev_channels, H, W)
#         fusion_feat = torch.cat([lidar_bev_feat, img_bev_feat], dim=1)   # (B, lidar_bev_channels+img_downsample_channels, H, W)
#         fusion_feat = self.fusion_conv(fusion_feat)                       # (B, out_channels, H, W)
#         return fusion_feat

#     def _batch_bilinear_sample(self, feat_map, u, v):
#         N = u.shape[0]
#         if N == 0:
#             return torch.empty(0, feat_map.shape[0], device=feat_map.device)
#         C, H, W = feat_map.shape
#         u_norm = 2.0 * u / (W - 1) - 1.0
#         v_norm = 2.0 * v / (H - 1) - 1.0
#         grid = torch.stack([u_norm, v_norm], dim=1).view(1, N, 1, 2)
#         feat_batch = feat_map.unsqueeze(0)
#         sampled = F.grid_sample(feat_batch, grid, mode='bilinear', align_corners=False)
#         sampled = sampled.squeeze(0).squeeze(2)
#         return sampled.permute(1, 0)

import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
import torch.nn.functional as F
import numpy as np

@MODELS.register_module()
class BEVPillarFusion(nn.Module):
    def __init__(self,
                 image_channels,          # 原始图像特征通道数，例如256
                 lidar_bev_channels,      # 点云BEV特征通道数
                 out_channels,            # 融合输出通道数
                 bev_h, bev_w,
                 point_cloud_range,
                 voxel_size,
                 num_samples=20,
                 attn_hidden_dim=64,
                 img_downsample_channels=32,  # 聚合后降维到的通道数
                 img_size=(256,408),
                 mode='normal',
                 **kwargs):
        super().__init__()
        self.image_channels = image_channels
        self.img_size = img_size
        self.lidar_bev_channels = lidar_bev_channels
        self.out_channels = out_channels
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.num_samples = num_samples
        self.attn_hidden_dim = attn_hidden_dim
        self.img_downsample_channels = img_downsample_channels
        self.mode = mode
        
        self.img_h = int(self.img_size[0]/8)
        self.img_w = int(self.img_size[1]/8)

        # 网格物理参数
        self.x_min, self.y_min, self.z_min = point_cloud_range[:3]
        self.x_max, self.y_max, self.z_max = point_cloud_range[3:]
        self.vx, self.vy, self.vz = voxel_size
        self.nx = int((self.x_max - self.x_min) / self.vx)
        self.ny = int((self.y_max - self.y_min) / self.vy)
        self.bev_h = bev_h if bev_h is not None else self.ny
        self.bev_w = bev_w if bev_w is not None else self.nx

        # 预计算网格中心坐标
        x_centers = torch.arange(self.bev_w, dtype=torch.float32) * self.vx + self.x_min + self.vx/2
        y_centers = torch.arange(self.bev_h, dtype=torch.float32) * self.vy + self.y_min + self.vy/2
        self.register_buffer('x_centers', x_centers)
        self.register_buffer('y_centers', y_centers)

        # 注意力聚合模块（输入为原始高维特征）
        self.attention = nn.Sequential(
            nn.Linear(image_channels, attn_hidden_dim),
            nn.ReLU(),
            nn.Linear(attn_hidden_dim, 1)
        )

        # 聚合后的降维层（高维 -> 低维）
        self.agg_downsample = nn.Linear(image_channels, img_downsample_channels)
        
        self.img_align_scale = nn.Parameter(torch.ones(1, img_downsample_channels, 1, 1))
        self.img_align_bias = nn.Parameter(torch.zeros(1, img_downsample_channels, 1, 1))

        # 融合卷积：输入通道 = 点云BEV通道 + 图像下采样通道
        # 根据模式设置融合卷积的输入通道数
        if self.mode == 'normal':
            concat_channels = lidar_bev_channels + img_downsample_channels
        elif self.mode == 'img_only':
            concat_channels = img_downsample_channels
        elif self.mode == 'radar_only':
            concat_channels = lidar_bev_channels
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        # concat_channels = lidar_bev_channels + img_downsample_channels
        self.fusion_conv = nn.Conv2d(concat_channels, out_channels, 1)

    def _vod_project_to_image(self, points, lidar2cam, cam_intrinsic, image_shape):
        """
        VOD 标准投影：点云 -> 相机坐标系 -> 图像像素坐标
        points: (N, 3) numpy array
        lidar2cam: (4, 4) 外参矩阵 (LiDAR -> Camera)
        cam_intrinsic: (3, 4) 内参投影矩阵
        image_shape: (H, W)
        """
        # 1. 齐次坐标
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        pts_homo = np.hstack([points[:, :3], ones])  # (N, 4)

        # 2. 变换到相机坐标系
        pts_cam = (lidar2cam @ pts_homo.T).T  # (N, 4)
        depth = pts_cam[:, 2]

        # 3. 投影到图像平面
        img_pts = (cam_intrinsic @ pts_cam.T).T  # (N, 3)
        u = img_pts[:, 0] / img_pts[:, 2]
        v = img_pts[:, 1] / img_pts[:, 2]

        # 4. 过滤有效点
        H, W = image_shape[:2]
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depth > 0)
        return u[valid].astype(np.int32), v[valid].astype(np.int32)

    def forward(self, lidar_bev_feat, points, img_feat, img_metas):
        B, C2, H, W = lidar_bev_feat.shape
        device = lidar_bev_feat.device

        # 处理图像特征维度（可能为5维，取第一个相机）
        if img_feat.dim() == 5:
            img_feat = img_feat[:, 0, ...]          # (B, C_orig, H_img, W_img)
        C_orig, H_img, W_img = img_feat.shape[1], img_feat.shape[2], img_feat.shape[3]
        
        # print(f"Img feat size: {C_orig, H_img, W_img}" )

        # 初始化图像BEV特征图（通道数为聚合后降维的通道数）
        img_bev_feat = torch.zeros(B, self.img_downsample_channels, H, W, device=device)

        # 预计算网格中心矩阵
        x_center_map = self.x_centers.view(1, -1).expand(H, -1)   # (H, W)
        y_center_map = self.y_centers.view(-1, 1).expand(-1, W)   # (H, W)

        for batch_idx in range(B):
            pts = points[batch_idx][:, :3]                      # (M, 3)
            
            if pts.shape[0] == 0:
                continue

            # ========== 合成总投影矩阵（支持数据增强） ==========
            # 1. 获取原始投影矩阵（LiDAR → 原始图像像素）
            lidar2img = np.array(img_metas[batch_idx]['lidar2img'][0])   # 4x4
            lidar2img = lidar2img.reshape(4, 4)

            # 2. 获取增强矩阵
            lidar_aug = np.eye(4)
            if 'lidar_aug_matrix' in img_metas[batch_idx]:
                lidar_aug = np.array(img_metas[batch_idx]['lidar_aug_matrix'])
            img_aug = np.eye(4)
            if 'img_aug_matrix' in img_metas[batch_idx]:
                img_aug = np.array(img_metas[batch_idx]['img_aug_matrix'][0])  # 第一个相机

            # 获取原始投影矩阵（未经过任何数据增强）
            if 'ori_lidar2img' in img_metas[batch_idx]:
                lidar2img_orig = np.array(img_metas[batch_idx]['ori_lidar2img'][0])
            else:
                lidar2cam = np.array(img_metas[batch_idx]['lidar2cam'][0])
                cam2img = np.array(img_metas[batch_idx]['cam2img'][0])
                if cam2img.shape == (4,4):
                    lidar2img_orig = cam2img @ lidar2cam
                else:
                    lidar2img_orig = cam2img @ lidar2cam
                    lidar2img_orig = np.vstack([lidar2img_orig, [0,0,0,1]])

            lidar_aug = np.eye(4)
            if 'lidar_aug_matrix' in img_metas[batch_idx]:
                lidar_aug = np.array(img_metas[batch_idx]['lidar_aug_matrix'])
            lidar_aug_inv = np.linalg.inv(lidar_aug)

            img_aug = np.eye(4)
            if 'img_aug_matrix' in img_metas[batch_idx]:
                img_aug = np.array(img_metas[batch_idx]['img_aug_matrix'][0])

            # ========== 修改点1：不再计算 total_proj，直接使用分步投影（与 DEBUG_VIS 内部一致） ==========
            # 以下为分步投影计算（numpy），得到增强图像坐标 u, v
            
            # 将 pts 转为 numpy 进行计算（点云数量少，开销可忽略）
            pts_np = pts.cpu().numpy()
            ones_np = np.ones((pts_np.shape[0], 1), dtype=np.float32)
            pts_aug_homo = np.hstack([pts_np, ones_np])          # (N,4)

            # 步骤1: 增强点云 -> 原始点云
            pts_orig_homo = pts_aug_homo @ lidar_aug_inv.T       # (N,4)
            # 步骤2: 原始点云 -> 原始图像坐标
            img_pts_raw = (lidar2img_orig @ pts_orig_homo.T).T   # (N,4)
            u_raw = img_pts_raw[:, 0] / img_pts_raw[:, 2]
            v_raw = img_pts_raw[:, 1] / img_pts_raw[:, 2]
            depth_raw = img_pts_raw[:, 2]
            # 步骤3: 原始图像坐标 -> 增强图像坐标
            uv1 = np.stack([u_raw, v_raw, np.ones_like(u_raw), np.ones_like(u_raw)], axis=1)  # (N,4)
            uv_aug = (img_aug @ uv1.T).T
            u = uv_aug[:, 0] / uv_aug[:, 2]
            v = uv_aug[:, 1] / uv_aug[:, 2]
            depth = depth_raw
            
            # print(f"Origin PTS: {pts_np}")
            # print(f"Mapping u: {u}")
            # print(f"Mapping v: {v}" )

            # 转为 tensor 并移到设备
            u = torch.from_numpy(u).float().to(device)
            v = torch.from_numpy(v).float().to(device)
            depth = torch.from_numpy(depth).float().to(device)
            # ========== 修改点1结束 ==========

            # ========== 调试可视化（保持不变） ==========
            DEBUG_VIS = False   # 训练时请改为 False
            if DEBUG_VIS:
                import cv2
                from mmengine.dist import get_rank, is_distributed
                if True:
                    print("\n========== STEP-BY-STEP VERIFICATION ==========")
                    img_meta = img_metas[batch_idx]

                    # 获取图像路径
                    img_path = None
                    for key in ['img_path', 'filename', 'ori_filename']:
                        if key in img_meta:
                            val = img_meta[key]
                            img_path = val[0] if isinstance(val, list) else val
                            break
                    if img_path is None:
                        print("ERROR: No image path found!")
                    else:
                        img_bgr = cv2.imread(img_path)
                        if img_bgr is None:
                            print(f"Failed to read image: {img_path}")
                        else:
                            H_orig, W_orig = img_bgr.shape[:2]
                            print(f"Original image: {img_path} ({H_orig}x{W_orig})")

                            # ---- 获取原始投影矩阵（优先 ori_lidar2img） ----
                            if 'ori_lidar2img' in img_meta:
                                lidar2img_orig = np.array(img_meta['ori_lidar2img'][0])
                            else:
                                lidar2cam = np.array(img_meta['lidar2cam'][0])
                                cam2img = np.array(img_meta['cam2img'][0])
                                if cam2img.shape == (4,4):
                                    lidar2img_orig = cam2img @ lidar2cam
                                else:
                                    lidar2img_orig = cam2img @ lidar2cam
                                    lidar2img_orig = np.vstack([lidar2img_orig, [0,0,0,1]])
                            print("lidar2img_orig shape:", lidar2img_orig.shape)

                            # ---- 获取增强矩阵 ----
                            lidar_aug = np.eye(4)
                            if 'lidar_aug_matrix' in img_meta:
                                lidar_aug = np.array(img_meta['lidar_aug_matrix'])
                            lidar_aug_inv = np.linalg.inv(lidar_aug)

                            img_aug = np.eye(4)
                            if 'img_aug_matrix' in img_meta:
                                img_aug = np.array(img_meta['img_aug_matrix'][0])
                            print("img_aug (first 3 rows):\n", img_aug[:3])

                            # ---- 点云数据 ----
                            pts_aug = points[batch_idx][:, :3].cpu().numpy()
                            ones = np.ones((pts_aug.shape[0], 1), dtype=np.float32)
                            pts_aug_homo = np.hstack([pts_aug, ones])        # (N,4)

                            # ---- 分步计算（原始投影 -> 图像增强） ----
                            # 步骤1: 增强点云 -> 原始坐标系 -> 原始图像坐标
                            pts_orig_homo = pts_aug_homo @ lidar_aug_inv.T   # (N,4)
                            img_pts_raw = (lidar2img_orig @ pts_orig_homo.T).T  # (N,4)
                            u_raw = img_pts_raw[:, 0] / img_pts_raw[:, 2]
                            v_raw = img_pts_raw[:, 1] / img_pts_raw[:, 2]

                            # 步骤2: 对原始图像坐标应用 img_aug 变换
                            uv1 = np.stack([u_raw, v_raw, np.ones_like(u_raw), np.ones_like(u_raw)], axis=1)  # (N,4)
                            uv_aug_homo = (img_aug @ uv1.T).T
                            u_step = uv_aug_homo[:, 0] / uv_aug_homo[:, 2]
                            v_step = uv_aug_homo[:, 1] / uv_aug_homo[:, 2]

                            # ---- 生成增强图像 ----
                            M_affine = img_aug[:2, [0,1,3]].astype(np.float32)
                            final_width, final_height = 408, 256
                            img_augmented = cv2.warpAffine(img_bgr, M_affine, (final_width, final_height))
                            cv2.imwrite('debug_augmented_image.jpg', img_augmented)
                            print("Saved debug_augmented_image.jpg (image only)")

                            # 过滤有效点（在增强图像范围内）
                            valid_step = (u_step >= 0) & (u_step < final_width) & (v_step >= 0) & (v_step < final_height)
                            print(f"Stepwise valid points: {np.sum(valid_step)}")

                            # 绘制绿色点（分步投影）
                            img_out = img_augmented.copy()
                            if np.sum(valid_step) > 0:
                                u_s = u_step[valid_step].astype(np.int32)
                                v_s = v_step[valid_step].astype(np.int32)
                                for ui, vi in zip(u_s, v_s):
                                    cv2.circle(img_out, (ui, vi), 2, (0, 255, 0), -1)
                            cv2.imwrite('debug_stepwise_proj.jpg', img_out)
                            print("Saved debug_stepwise_proj.jpg (green=stepwise)")

                            # 额外保存原始投影（参考）
                            img_raw_out = img_bgr.copy()
                            valid_raw = (u_raw >= 0) & (u_raw < W_orig) & (v_raw >= 0) & (v_raw < H_orig)
                            if np.sum(valid_raw) > 0:
                                u_r = u_raw[valid_raw].astype(np.int32)
                                v_r = v_raw[valid_raw].astype(np.int32)
                                for ui, vi in zip(u_r, v_r):
                                    cv2.circle(img_raw_out, (ui, vi), 2, (0, 0, 255), -1)
                            cv2.imwrite('debug_raw_proj_ref.jpg', img_raw_out)
                            print("Saved debug_raw_proj_ref.jpg (raw projection, should be correct)")
                            
                            print(f"DEBUG Origin PTS: {pts_aug}")
                            print(f"DEBUG Mapping u: {u_step}")
                            print(f"DEBUG Mapping v: {v_step}" )
                            print(f"Max u: {max(u_step)}")
                            print(f"Max v: {max(v_step)}")

                            input("Press Enter to continue...")
                    print("========== DEBUG END ==========\n")

            # ========== 修改点2：特征投影部分使用上面计算的 u, v（分步投影结果） ==========
            # 不再使用 total_proj，直接使用分步得到的 u, v 和 depth
            # 网格分配和过滤
            
            # 计算网格坐标（基于原始点云坐标）
            grid_x = ((pts[:, 0] - self.x_min) / self.vx).long()
            grid_y = ((pts[:, 1] - self.y_min) / self.vy).long()
            valid = (grid_x >= 0) & (grid_x < W) & (grid_y >= 0) & (grid_y < H)
            grid_x = grid_x[valid]
            grid_y = grid_y[valid]
            pts_valid = pts[valid]
            u = u[valid]
            v = v[valid]
            depth = depth[valid]

            if pts_valid.shape[0] == 0:
                continue

            # 投影有效性过滤（在图像特征图范围内）
            valid_proj = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img) & (depth > 0)
            u = u[valid_proj]
            v = v[valid_proj]
            grid_x = grid_x[valid_proj]
            grid_y = grid_y[valid_proj]
            pts_valid = pts_valid[valid_proj]

            if pts_valid.shape[0] == 0:
                continue

            # 批量双线性采样（得到高维特征）
            img_feat_b = img_feat[batch_idx]                     # (C_orig, H_img, W_img)
            sampled_feat = self._batch_bilinear_sample(img_feat_b, u, v)  # (K, C_orig)

            # 按网格分组
            grid_indices = grid_y * W + grid_x
            unique_grids, inverse, counts = torch.unique(grid_indices, return_inverse=True, return_counts=True)

            selected_per_grid = {}
            for idx, gid in enumerate(unique_grids):
                mask = inverse == idx
                pts_in_grid = pts_valid[mask]
                feats_in_grid = sampled_feat[mask]
                num_pts = pts_in_grid.shape[0]
                if num_pts >= self.num_samples:
                    chosen = np.random.choice(num_pts, self.num_samples, replace=False)
                    selected_per_grid[gid.item()] = (pts_in_grid[chosen], feats_in_grid[chosen])
                else:
                    selected_per_grid[gid.item()] = (pts_in_grid, feats_in_grid)

            # ---------- 虚拟点生成与投影（也使用分步投影） ----------
            need_virtual = set()
            for gid, (pts_g, _) in selected_per_grid.items():
                if pts_g.shape[0] < self.num_samples:
                    need_virtual.add(gid)

            virtual_points_dict = {}
            if need_virtual:
                # 获取需要补齐的网格中心坐标（numpy）
                gx_list = [gid % W for gid in need_virtual]
                gy_list = [gid // W for gid in need_virtual]
                x_centers_needed = x_center_map[gy_list, gx_list].cpu().numpy()
                y_centers_needed = y_center_map[gy_list, gx_list].cpu().numpy()

                K_max = self.num_samples
                ratios = np.linspace(1/(K_max+1), K_max/(K_max+1), K_max)
                z_virt = self.z_min + ratios * (self.z_max - self.z_min)
                x_expand = np.expand_dims(x_centers_needed, axis=1).repeat(K_max, axis=1)
                y_expand = np.expand_dims(y_centers_needed, axis=1).repeat(K_max, axis=1)
                z_expand = np.tile(z_virt, (len(need_virtual), 1))
                virtual_xyz = np.stack([x_expand, y_expand, z_expand], axis=-1)  # (N_g, K_max, 3)

                N_g = len(need_virtual)
                virtual_flat = virtual_xyz.reshape(-1, 3)
                ones_v = np.ones((virtual_flat.shape[0], 1), dtype=np.float32)
                v_homo = np.hstack([virtual_flat, ones_v])

                # 分步投影虚拟点（与真实点相同）
                v_orig_homo = v_homo @ lidar_aug_inv.T
                v_img_raw = (lidar2img_orig @ v_orig_homo.T).T
                v_u_raw = v_img_raw[:, 0] / v_img_raw[:, 2]
                v_v_raw = v_img_raw[:, 1] / v_img_raw[:, 2]
                v_uv1 = np.stack([v_u_raw, v_v_raw, np.ones_like(v_u_raw), np.ones_like(v_u_raw)], axis=1)
                v_uv_aug = (img_aug @ v_uv1.T).T
                v_u = v_uv_aug[:, 0] / v_uv_aug[:, 2]
                v_v = v_uv_aug[:, 1] / v_uv_aug[:, 2]
                v_depth = v_img_raw[:, 2]

                v_u = torch.from_numpy(v_u).float().to(device)
                v_v = torch.from_numpy(v_v).float().to(device)
                v_depth = torch.from_numpy(v_depth).float().to(device)

                v_valid = (v_u >= 0) & (v_u < W_img) & (v_v >= 0) & (v_v < H_img) & (v_depth > 0)
                v_feat = self._batch_bilinear_sample(img_feat_b, v_u, v_v)
                v_feat[~v_valid] = 0.0
                v_feat = v_feat.reshape(N_g, K_max, C_orig)

                for i, gid in enumerate(need_virtual):
                    real_cnt = selected_per_grid[gid][0].shape[0]
                    need_cnt = self.num_samples - real_cnt
                    if need_cnt > 0:
                        virtual_points_dict[gid] = v_feat[i, :need_cnt, :]
                    else:
                        virtual_points_dict[gid] = torch.empty(0, C_orig, device=device)

            # ---------- 3. 每个网格：聚合高维特征 -> 降维 -> 填入BEV图 ----------
            for gy in range(H):
                for gx in range(W):
                    gid = gy * W + gx
                    real_pts, real_feat = selected_per_grid.get(gid, (torch.empty(0,3,device=device), torch.empty(0,C_orig,device=device)))
                    real_cnt = real_feat.shape[0]
                    if real_cnt == 0 and gid not in virtual_points_dict:
                        continue

                    all_feats = [real_feat]
                    if gid in virtual_points_dict:
                        virt_feat = virtual_points_dict[gid]
                        if virt_feat.shape[0] > 0:
                            all_feats.append(virt_feat)
                    all_feats = torch.cat(all_feats, dim=0)          # (total, C_orig)
                    total = all_feats.shape[0]
                    if total < self.num_samples:
                        pad = torch.zeros(self.num_samples - total, C_orig, device=device)
                        all_feats = torch.cat([all_feats, pad], dim=0)
                    elif total > self.num_samples:
                        idx = torch.randperm(total)[:self.num_samples]
                        all_feats = all_feats[idx]

                    # 注意力聚合（在高维空间）
                    attn_weights = self.attention(all_feats)        # (N, 1)
                    attn_weights = torch.softmax(attn_weights, dim=0)
                    agg_feat_high = (all_feats * attn_weights).sum(dim=0)  # (C_orig,)

                    # 降维到低维
                    agg_feat_low = self.agg_downsample(agg_feat_high)     # (img_downsample_channels,)

                    img_bev_feat[batch_idx, :, gy, gx] = agg_feat_low
                    
        img_bev_feat = img_bev_feat * self.img_align_scale + self.img_align_bias

        # ---------- 4. 融合：直接拼接点云BEV特征和图像BEV特征，再卷积输出 ----------
        # ========== 根据模式进行融合 ==========
        if self.mode == 'normal':
            fusion_input = torch.cat([lidar_bev_feat, img_bev_feat], dim=1)
        elif self.mode == 'img_only':
            fusion_input = img_bev_feat
        elif self.mode == 'radar_only':
            fusion_input = lidar_bev_feat
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        fusion_feat = self.fusion_conv(fusion_input)
        return fusion_feat

    # def _batch_bilinear_sample(self, feat_map, u, v):
    #     N = u.shape[0]
    #     if N == 0:
    #         return torch.empty(0, feat_map.shape[0], device=feat_map.device)
    #     C, H, W = feat_map.shape
    #     u_norm = 2.0 * u / (W - 1) - 1.0
    #     v_norm = 2.0 * v / (H - 1) - 1.0
    #     grid = torch.stack([u_norm, v_norm], dim=1).view(1, N, 1, 2)
    #     feat_batch = feat_map.unsqueeze(0)
    #     sampled = F.grid_sample(feat_batch, grid, mode='bilinear', align_corners=False)
    #     sampled = sampled.squeeze(0).squeeze(2)
    #     return sampled.permute(1, 0)
    
    def _batch_bilinear_sample(self, feat_map, u, v):
        N = u.shape[0]
        if N == 0:
            return torch.empty(0, feat_map.shape[0], device=feat_map.device)
        C, H, W = feat_map.shape
        u_int = torch.floor(u/8).long().clamp(0, self.img_w-1)
        v_int = torch.floor(v/8).long().clamp(0, self.img_h-1)
        sampled = feat_map[:, v_int, u_int]
        return sampled.permute(1, 0)

from mmdet3d.models.voxel_encoders import SpatialChannelFusion

@MODELS.register_module()
class YOZPillarFusion(nn.Module):
    """
    YOZ 平面图像特征提取 + 注意力融合到 BEV 空间。
    输入：
        lidar_bev_feat: (B, C_bev, H_bev, W_bev)  BEV 特征图（主特征）
        points: list of (N, 3) 点云坐标
        img_feat: (B, C_img, H_img, W_img) 或 (B, N_cam, C_img, H_img, W_img)
        img_metas: 元信息，包含投影矩阵和增强矩阵
    输出：
        fusion_feat: (B, out_channels, H_bev, W_bev)
    """
    def __init__(self,
                 image_channels,          # 图像特征通道数
                 lidar_bev_channels,      # BEV 特征通道数
                 out_channels,            # 融合输出通道数
                 bev_h, bev_w,            # BEV 网格尺寸 (Y, X)
                 point_cloud_range,       # [x_min, y_min, z_min, x_max, y_max, z_max]
                 voxel_size,              # [vx, vy, vz] 用于 BEV 网格，YOZ 网格独立定义
                 num_samples=20,          # 每个网格固定采样点数
                 attn_hidden_dim=64,      # 注意力聚合 MLP 隐藏维度
                 img_downsample_channels=32,  # 聚合后降维通道数
                 yoz_h=0,
                 img_size=(256,408),      # 原始图像尺寸 (H, W)
                 d_model=128,             # 跨模态注意力模型维度
                 n_heads=4,               # 注意力头数
                 dropout=0.1,
                 **kwargs):
        super().__init__()
        
        yoz_y_bound = [point_cloud_range[1], point_cloud_range[4]]
        yoz_z_bound = [point_cloud_range[2], point_cloud_range[5]]
        self.yoz_y_min, self.yoz_y_max = yoz_y_bound
        self.yoz_z_min, self.yoz_z_max = yoz_z_bound

        self.yoz_dy = voxel_size[1]
        self.yoz_dz = voxel_size[2]
        if yoz_h == 0:
            self.yoz_h = max(1, int((self.yoz_y_max - self.yoz_y_min) / self.yoz_dy))
        else:
            self.yoz_h = yoz_h
        self.yoz_w = max(1, int((self.yoz_z_max - self.yoz_z_min) / self.yoz_dz))
        
        self.image_channels = image_channels
        self.img_size = img_size
        self.lidar_bev_channels = lidar_bev_channels
        self.out_channels = out_channels
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.num_samples = num_samples
        self.attn_hidden_dim = attn_hidden_dim
        self.img_downsample_channels = img_downsample_channels

        # 图像特征图尺寸（下采样8倍）
        self.img_h = int(self.img_size[0] / 8)
        self.img_w = int(self.img_size[1] / 8)

        # BEV 网格物理参数（用于计算点云的 BEV 网格坐标，但本模块不直接使用，留给外部）
        self.x_min, self.y_min, self.z_min = point_cloud_range[:3]
        self.x_max, self.y_max, self.z_max = point_cloud_range[3:]
        self.vx, self.vy, self.vz = voxel_size
        self.nx = int((self.x_max - self.x_min) / self.vx)
        self.ny = int((self.y_max - self.y_min) / self.vy)
        self.bev_h = bev_h if bev_h is not None else self.ny
        self.bev_w = bev_w if bev_w is not None else self.nx

        # 预计算 YOZ 网格中心坐标（Y 和 Z）
        y_centers = torch.arange(self.yoz_h, dtype=torch.float32) * self.yoz_dy + self.yoz_y_min + self.yoz_dy/2
        z_centers = torch.arange(self.yoz_w, dtype=torch.float32) * self.yoz_dz + self.yoz_z_min + self.yoz_dz/2
        self.register_buffer('yoz_y_centers', y_centers)
        self.register_buffer('yoz_z_centers', z_centers)

        # 预计算网格中心矩阵（用于虚拟点生成）
        self.register_buffer('yoz_y_center_map', y_centers.view(-1, 1).expand(self.yoz_h, self.yoz_w))
        self.register_buffer('yoz_z_center_map', z_centers.view(1, -1).expand(self.yoz_h, self.yoz_w))

        # 注意力聚合模块（对每个网格内的 N 个采样点特征进行聚合）
        self.point_attention = nn.Sequential(
            nn.Linear(image_channels, attn_hidden_dim),
            nn.ReLU(),
            nn.Linear(attn_hidden_dim, 1)
        )
        # 降维层
        self.agg_downsample = nn.Linear(image_channels, img_downsample_channels)
        
        self.img_align_scale = nn.Parameter(torch.ones(1, img_downsample_channels, 1, 1))
        self.img_align_bias = nn.Parameter(torch.zeros(1, img_downsample_channels, 1, 1))

        # 跨模态注意力融合（BEV 主特征 + YOZ 特征）
        self.cross_attention = SpatialChannelFusion(
            in_ch_bev=lidar_bev_channels,
            in_ch_side=img_downsample_channels,
            out_ch=out_channels,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_len=max(self.bev_h, self.yoz_w)
        )

        # 最终卷积（可选，如果 cross_attention 输出通道已经是 out_channels 则不需要）
        # 这里 cross_attention 输出 out_channels，直接返回

    def _batch_sample_feat(self, feat_map, u, v):
        """最近邻采样（图像特征图下采样8倍）"""
        N = u.shape[0]
        if N == 0:
            return torch.empty(0, feat_map.shape[0], device=feat_map.device)
        C, H, W = feat_map.shape
        u_int = torch.floor(u / 8).long().clamp(0, self.img_w - 1)
        v_int = torch.floor(v / 8).long().clamp(0, self.img_h - 1)
        sampled = feat_map[:, v_int, u_int]   # (C, N)
        return sampled.permute(1, 0)          # (N, C)

    def forward(self, lidar_bev_feat, points, img_feat, img_metas):
        B, C_bev, H_bev, W_bev = lidar_bev_feat.shape
        device = lidar_bev_feat.device

        # 处理图像特征维度
        if img_feat.dim() == 5:
            img_feat = img_feat[:, 0, ...]          # (B, C_img, H_img, W_img)
        C_img, H_img, W_img = img_feat.shape[1], img_feat.shape[2], img_feat.shape[3]

        # 初始化 YOZ 特征图
        yoz_feat = torch.zeros(B, self.img_downsample_channels, self.yoz_h, self.yoz_w, device=device)

        # 预计算 YOZ 中心坐标矩阵（numpy 版本用于快速索引）
        yoz_y_center_np = self.yoz_y_center_map.cpu().numpy()
        yoz_z_center_np = self.yoz_z_center_map.cpu().numpy()

        for batch_idx in range(B):
            pts = points[batch_idx][:, :3]          # (M, 3)
            if pts.shape[0] == 0:
                continue

            # ---------- 获取投影矩阵 ----------
            # 原始投影矩阵（LiDAR → 原始图像像素）
            if 'ori_lidar2img' in img_metas[batch_idx]:
                lidar2img_orig = np.array(img_metas[batch_idx]['ori_lidar2img'][0])
            else:
                lidar2cam = np.array(img_metas[batch_idx]['lidar2cam'][0])
                cam2img = np.array(img_metas[batch_idx]['cam2img'][0])
                if cam2img.shape == (4, 4):
                    lidar2img_orig = cam2img @ lidar2cam
                else:
                    lidar2img_orig = cam2img @ lidar2cam
                    lidar2img_orig = np.vstack([lidar2img_orig, [0, 0, 0, 1]])

            # 数据增强矩阵
            lidar_aug = np.eye(4)
            if 'lidar_aug_matrix' in img_metas[batch_idx]:
                lidar_aug = np.array(img_metas[batch_idx]['lidar_aug_matrix'])
            lidar_aug_inv = np.linalg.inv(lidar_aug)

            img_aug = np.eye(4)
            if 'img_aug_matrix' in img_metas[batch_idx]:
                img_aug = np.array(img_metas[batch_idx]['img_aug_matrix'][0])

            # ---------- 将点云分配到 YOZ 网格 ----------
            # 计算每个点所属的 YOZ 网格索引
            y_indices = ((pts[:, 1] - self.yoz_y_min) / self.yoz_dy).long()
            z_indices = ((pts[:, 2] - self.yoz_z_min) / self.yoz_dz).long()
            valid_yoz = (y_indices >= 0) & (y_indices < self.yoz_h) & (z_indices >= 0) & (z_indices < self.yoz_w)
            if not valid_yoz.any():
                continue

            # 只保留有效点
            pts_valid = pts[valid_yoz]
            y_idx = y_indices[valid_yoz]
            z_idx = z_indices[valid_yoz]

            # 投影有效点到图像（分步投影，支持增强）
            pts_np = pts_valid.cpu().numpy()
            ones_np = np.ones((pts_np.shape[0], 1), dtype=np.float32)
            pts_aug_homo = np.hstack([pts_np, ones_np])                 # (K,4)

            pts_orig_homo = pts_aug_homo @ lidar_aug_inv.T              # (K,4)
            img_pts_raw = (lidar2img_orig @ pts_orig_homo.T).T          # (K,4)
            u_raw = img_pts_raw[:, 0] / img_pts_raw[:, 2]
            v_raw = img_pts_raw[:, 1] / img_pts_raw[:, 2]
            depth_raw = img_pts_raw[:, 2]

            uv1 = np.stack([u_raw, v_raw, np.ones_like(u_raw), np.ones_like(u_raw)], axis=1)
            uv_aug = (img_aug @ uv1.T).T
            u = uv_aug[:, 0] / uv_aug[:, 2]
            v = uv_aug[:, 1] / uv_aug[:, 2]
            depth = depth_raw

            u = torch.from_numpy(u).float().to(device)
            v = torch.from_numpy(v).float().to(device)
            depth = torch.from_numpy(depth).float().to(device)

            # 过滤投影有效的点（在图像特征图范围内）
            valid_proj = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img) & (depth > 0)
            if not valid_proj.any():
                continue
            u = u[valid_proj]
            v = v[valid_proj]
            y_idx = y_idx[valid_proj]
            z_idx = z_idx[valid_proj]
            pts_valid = pts_valid[valid_proj]

            # 采样图像特征
            img_feat_b = img_feat[batch_idx]                # (C_img, H_img, W_img)
            sampled_feat = self._batch_sample_feat(img_feat_b, u, v)   # (K, C_img)

            # 按 YOZ 网格分组
            grid_ids = y_idx * self.yoz_w + z_idx
            unique_gids, inverse, counts = torch.unique(grid_ids, return_inverse=True, return_counts=True)

            # 存储每个网格的真实点特征（未补齐）
            grid_real_feats = {}
            for idx, gid in enumerate(unique_gids):
                mask = inverse == idx
                feats = sampled_feat[mask]
                num_pts = feats.shape[0]
                if num_pts >= self.num_samples:
                    chosen = torch.randperm(num_pts, device=device)[:self.num_samples]
                    feats = feats[chosen]
                grid_real_feats[gid.item()] = feats   # 可能少于 num_samples

            # ---------- 虚拟点生成与投影 ----------
            need_virtual = {gid for gid, feats in grid_real_feats.items() if feats.shape[0] < self.num_samples}
            virtual_feats_dict = {}
            if need_virtual:
                # 获取需要补齐的网格的 (y, z) 中心坐标
                gid_list = list(need_virtual)
                gy_list = [gid // self.yoz_w for gid in gid_list]
                gz_list = [gid % self.yoz_w for gid in gid_list]
                y_centers = yoz_y_center_np[gy_list, gz_list]   # (N_g,)
                z_centers = yoz_z_center_np[gy_list, gz_list]   # (N_g,)

                K_max = self.num_samples
                # 沿 X 轴均匀采样
                ratios = np.linspace(1/(K_max+1), K_max/(K_max+1), K_max)
                x_virt = ratios * self.x_max   # (K_max,)
                # 生成虚拟点坐标 (N_g, K_max, 3)
                x_expand = np.tile(x_virt, (len(need_virtual), 1))
                y_expand = np.tile(y_centers[:, None], (1, K_max))
                z_expand = np.tile(z_centers[:, None], (1, K_max))
                virtual_xyz = np.stack([x_expand, y_expand, z_expand], axis=-1)   # (N_g, K_max, 3)

                # 投影虚拟点
                virtual_flat = virtual_xyz.reshape(-1, 3)
                ones_v = np.ones((virtual_flat.shape[0], 1), dtype=np.float32)
                v_homo = np.hstack([virtual_flat, ones_v])
                v_orig_homo = v_homo @ lidar_aug_inv.T
                v_img_raw = (lidar2img_orig @ v_orig_homo.T).T
                v_u_raw = v_img_raw[:, 0] / v_img_raw[:, 2]
                v_v_raw = v_img_raw[:, 1] / v_img_raw[:, 2]
                v_uv1 = np.stack([v_u_raw, v_v_raw, np.ones_like(v_u_raw), np.ones_like(v_u_raw)], axis=1)
                v_uv_aug = (img_aug @ v_uv1.T).T
                v_u = v_uv_aug[:, 0] / v_uv_aug[:, 2]
                v_v = v_uv_aug[:, 1] / v_uv_aug[:, 2]
                v_depth = v_img_raw[:, 2]

                v_u = torch.from_numpy(v_u).float().to(device)
                v_v = torch.from_numpy(v_v).float().to(device)
                v_depth = torch.from_numpy(v_depth).float().to(device)

                v_valid = (v_u >= 0) & (v_u < W_img) & (v_v >= 0) & (v_v < H_img) & (v_depth > 0)
                v_feat = self._batch_sample_feat(img_feat_b, v_u, v_v)   # (N_g*K_max, C_img)
                v_feat[~v_valid] = 0.0
                v_feat = v_feat.reshape(len(need_virtual), K_max, C_img)

                for i, gid in enumerate(gid_list):
                    real_cnt = grid_real_feats[gid].shape[0]
                    need_cnt = self.num_samples - real_cnt
                    if need_cnt > 0:
                        virtual_feats_dict[gid] = v_feat[i, :need_cnt, :]
                    else:
                        virtual_feats_dict[gid] = torch.empty(0, C_img, device=device)

            # ---------- 聚合每个 YOZ 网格的特征 ----------
            # 遍历所有 YOZ 网格（包括空网格）
            for gy in range(self.yoz_h):
                for gz in range(self.yoz_w):
                    gid = gy * self.yoz_w + gz
                    real_feat = grid_real_feats.get(gid, torch.empty(0, C_img, device=device))
                    virt_feat = virtual_feats_dict.get(gid, torch.empty(0, C_img, device=device))

                    all_feats = []
                    if real_feat.numel() > 0:
                        all_feats.append(real_feat)
                    if virt_feat.numel() > 0:
                        all_feats.append(virt_feat)
                    if not all_feats:
                        continue   # 空网格，特征保持为0

                    all_feats = torch.cat(all_feats, dim=0)   # (total, C_img)
                    total = all_feats.shape[0]
                    if total < self.num_samples:
                        pad = torch.zeros(self.num_samples - total, C_img, device=device)
                        all_feats = torch.cat([all_feats, pad], dim=0)
                    elif total > self.num_samples:
                        idx = torch.randperm(total, device=device)[:self.num_samples]
                        all_feats = all_feats[idx]

                    # 注意力聚合
                    attn_weights = self.point_attention(all_feats)   # (N,1)
                    attn_weights = torch.softmax(attn_weights, dim=0)
                    agg_high = (all_feats * attn_weights).sum(dim=0)  # (C_img,)
                    agg_low = self.agg_downsample(agg_high)           # (img_downsample_channels,)
                    yoz_feat[batch_idx, :, gy, gz] = agg_low
                    
        yoz_feat = yoz_feat * self.img_align_scale + self.img_align_bias

        # ---------- 跨模态注意力融合（复用原始 SpatialChannelFusion） ----------
        # BEV 特征: (B, C_bev, H_bev, W_bev) -> (B, H_bev, W_bev, C_bev) -> (B, Y, X, C_bev)
        bev_feat_4d = lidar_bev_feat.permute(0, 2, 3, 1)   # (B, Y, X, C_bev)

        # YOZ 特征: (B, C_side, H_yoz, W_yoz) -> (B, H_yoz, W_yoz, C_side) -> (B, Y, Z, C_side)
        yoz_feat_4d = yoz_feat.permute(0, 2, 3, 1)         # (B, Y, Z, C_side)

        fused_list = []
        for b in range(B):
            bev = bev_feat_4d[b]          # (Y, X, C_bev)
            side = yoz_feat_4d[b]         # (Y, Z, C_side)

            # 转置以符合原始 SpatialChannelFusion 的输入格式
            bev_t = bev.permute(1, 0, 2)  # (X, Y, C_bev)
            side_t = side.permute(1, 0, 2)  # (Z, Y, C_side)

            # 调用原始融合模块
            fused_t = self.cross_attention(bev_t, side_t)  # 输出 (X, Y, out_ch)

            # 转置回 (Y, X, out_ch)
            fused = fused_t.permute(1, 0, 2)               # (Y, X, out_ch)
            fused_list.append(fused.unsqueeze(0))

        fused = torch.cat(fused_list, dim=0)               # (B, Y, X, out_ch)
        fused = fused.permute(0, 3, 1, 2)                  # (B, out_ch, Y, X)
        return fused