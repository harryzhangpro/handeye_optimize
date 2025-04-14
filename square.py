
import open3d as o3d
import numpy as np
import os
import glob
import argparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

# ================== 命令行参数处理 ==================
parser = argparse.ArgumentParser(description="逐帧拟合球并打印报告")
parser.add_argument("--base_dir", type=str, default="./0411", help="点云文件夹路径")
parser.add_argument("--record_file", type=str, default="aubo_record_2025-04-11_22-15-34.txt", help="机械臂记录文件")
parser.add_argument("--calib_file", type=str, default="cal.txt", help="手眼标定文件")
args = parser.parse_args()

base_dir = args.base_dir
record_file = args.record_file
calib_file = args.calib_file

# ================== 加载手眼标定矩阵 ==================
with open(calib_file, "r") as f:
    content = f.read()
    end_T_cam = eval(content.split('=')[1].strip())

# ================== 加载Aubo记录位姿 ==================
pose_list = []
with open(record_file, "r") as f:
    lines = f.readlines()
    for i in range(len(lines)):
        if "Position" in lines[i]:
            pos = eval(lines[i].split(":")[1])
            ori = eval(lines[i+1].split(":")[1])
            pose_list.append((pos, ori))

# ================== 加载点云文件路径 ==================
pcd_files = sorted(glob.glob(os.path.join(base_dir, "point_cloud_*.pcd")))

def pose_to_matrix(position, quat_xyzw):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = position
    return T

def fit_sphere(points):
    def residuals(params, xyz):
        cx, cy, cz, r = params
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        return np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2) - r

    center_est = points.mean(axis=0)
    radius_est = np.mean(np.linalg.norm(points - center_est, axis=1))
    initial_guess = np.hstack((center_est, radius_est))

    result = least_squares(residuals, initial_guess, args=(points,))
    return result.x, residuals(result.x, points)

def print_fitting_report(residuals):
    abs_residuals = np.abs(residuals)
    print("--- 点云拟合球面误差报告（单位：mm）---")
    print("最大误差    : {:.3f} mm".format(np.max(abs_residuals)))
    print("最小误差    : {:.3f} mm".format(np.min(abs_residuals)))
    print("平均误差    : {:.3f} mm".format(np.mean(abs_residuals)))
    print("标准差      : {:.3f} mm".format(np.std(abs_residuals)))
    print("RMSE（均方根）: {:.3f} mm".format(np.sqrt(np.mean(abs_residuals**2))))
    print("--------------------------------------\n")

# ================== 逐帧处理并拟合球体 ==================
visual_geoms = []
for i, pcd_file in enumerate(pcd_files):
    print(f"\n处理第 {i} 帧: {os.path.basename(pcd_file)}")
    pcd = o3d.io.read_point_cloud(pcd_file)
    pcd.scale(0.001, center=(0, 0, 0))  # mm -> m

    pos, quat = pose_list[i]
    T_base_to_end = pose_to_matrix(pos, quat)
    T_base_to_cam = T_base_to_end @ end_T_cam
    pcd.transform(T_base_to_cam)

    points = np.asarray(pcd.points)
    cx, cy, cz, r, residuals = *fit_sphere(points)[0], fit_sphere(points)[1]

    # 转回 mm 单位输出
    print("球心坐标: ({:.2f}, {:.2f}, {:.2f}) mm".format(cx * 1000, cy * 1000, cz * 1000))
    print("球的直径: {:.2f} mm".format(2 * r * 1000))
    print_fitting_report(residuals * 1000)

    # 可视化用：球面点云
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x = cx + r * np.cos(u) * np.sin(v)
    y = cy + r * np.sin(u) * np.sin(v)
    z = cz + r * np.cos(v)
    sphere_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    sphere_pcd = o3d.geometry.PointCloud()
    sphere_pcd.points = o3d.utility.Vector3dVector(sphere_points)
    sphere_pcd.paint_uniform_color([1, 0, 0])  # 红色球

    pcd.paint_uniform_color([0, 1, 0])  # 绿色点云

    visual_geoms += [pcd, sphere_pcd]

# ================== 可视化所有结果 ==================
print("\n✅ 所有点云和拟合球处理完成，正在显示...")
o3d.visualization.draw_geometries(visual_geoms)
