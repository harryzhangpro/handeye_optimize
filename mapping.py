import open3d as o3d
import numpy as np
import os
import glob
import re
from scipy.spatial.transform import Rotation as R

# ================== 配置路径 ==================
base_dir = "./0411"  # 点云文件夹
record_file = "aubo_record_2025-04-11_22-15-34.txt"
calib_file = "cal5.txt"

# ================== 加载手眼标定矩阵 ==================
print("加载手眼标定矩阵...")
with open(calib_file, "r") as f:
    content = f.read()
    end_T_cam = eval(content.split('=')[1].strip())
print("end_T_cam =\n", end_T_cam)

# ================== 加载Aubo记录位姿 ==================
print("加载Aubo记录位姿...")
pose_list = []
with open(record_file, "r") as f:
    lines = f.readlines()

    for i in range(len(lines)):
        if "Position" in lines[i]:
            pos = eval(lines[i].split(":")[1])
            ori = eval(lines[i+1].split(":")[1])
            pose_list.append((pos, ori))
print(f"共读取到 {len(pose_list)} 个位姿记录")

# ================== 点云和姿态文件匹配 ==================
pcd_files = sorted(glob.glob(os.path.join(base_dir, "point_cloud_*.pcd")))
print(f"共检测到 {len(pcd_files)} 个点云文件")

if len(pcd_files) != len(pose_list):
    print("❌ 点云数量和姿态数量不一致！请检查数据是否缺失。")
    exit()

# ================== 生成每一帧的变换矩阵 ==================
def pose_to_matrix(position, quat_xyzw):
    """从位置和四元数 (xyzw) 构造变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()  # Scipy 默认 xyzw 顺序
    T[:3, 3] = position
    return T

# ================== 拼接点云 ==================
print("开始拼接点云...")
combined_pcd = o3d.geometry.PointCloud()

for i, pcd_file in enumerate(pcd_files):
    print(f"\n处理第 {i} 帧: {os.path.basename(pcd_file)}")
    pcd = o3d.io.read_point_cloud(pcd_file)

    # 在加载点云后，加上缩放
    pcd.scale(0.001, center=(0, 0, 0))  # 从 mm 缩放到 m
    
    # 获取对应的机器人位姿
    pos, quat = pose_list[i]
    print(f"Position: {pos}")
    print(f"Orientation (xyzw): {quat}")
    
    T_base_to_end = pose_to_matrix(pos, quat)
    print("T_base_to_end =\n", T_base_to_end)

    T_base_to_cam = T_base_to_end @ end_T_cam
    print("T_base_to_cam =\n", T_base_to_cam)

    # 应用变换
    pcd.transform(T_base_to_cam)
    
    # 累积点云
    combined_pcd += pcd

print("\n✅ 点云拼接完成，开始可视化...")
o3d.visualization.draw_geometries([combined_pcd])