from pypcd import pypcd
import numpy as np
import open3d as o3d

# 读取 pcd 文件
pc = pypcd.PointCloud.from_path("./0411/point_cloud_00000.pcd")

# 提取 x, y, z 并过滤无效值
points = np.stack([pc.pc_data['x'], pc.pc_data['y'], pc.pc_data['z']], axis=-1)
points = points[np.isfinite(points).all(axis=1)]  # 去除 NaN / inf

# 如果你需要颜色也可以从 pc.pc_data['rgba'] 中解码颜色

# 转为 Open3D 点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 可视化
print(f"有效点数：{len(pcd.points)}")
o3d.visualization.draw_geometries([pcd])