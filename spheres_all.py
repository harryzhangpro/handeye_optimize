
import open3d as o3d
import numpy as np
import os
import glob
import argparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

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
    print("--- 拟合误差报告（单位：mm）---")
    print("最大误差    : {:.3f} mm".format(np.max(abs_residuals)))
    print("最小误差    : {:.3f} mm".format(np.min(abs_residuals)))
    print("平均误差    : {:.3f} mm".format(np.mean(abs_residuals)))
    print("标准差      : {:.3f} mm".format(np.std(abs_residuals)))
    print("RMSE（均方根）: {:.3f} mm".format(np.sqrt(np.mean(abs_residuals**2))))
    print("--------------------------------")

def main():
    parser = argparse.ArgumentParser(description="逐帧拟合球体并评估误差")
    parser.add_argument("--base_dir", type=str, default="./0411", help="点云文件夹路径")
    parser.add_argument("--record_file", type=str, default="aubo_record_2025-04-11_22-15-34.txt", help="机械臂位姿文件")
    parser.add_argument("--calib_file", type=str, default="cal4.txt", help="手眼标定文件")
    args = parser.parse_args()

    # === 加载标定矩阵 ===
    with open(args.calib_file, "r") as f:
        content = f.read()
        end_T_cam = eval(content.split('=')[1].strip())

    # === 加载机械臂位姿 ===
    pose_list = []
    with open(args.record_file, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if "Position" in lines[i]:
                pos = eval(lines[i].split(":")[1])
                ori = eval(lines[i+1].split(":")[1])
                pose_list.append((pos, ori))

    # === 加载点云文件 ===
    pcd_files = sorted(glob.glob(os.path.join(args.base_dir, "point_cloud_*.pcd")))
    visual_geoms = []
    centers_mm = []

    for i, pcd_file in enumerate(pcd_files):
        print(f"\n处理第 {i} 帧: {os.path.basename(pcd_file)}")
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcd.scale(0.001, center=(0, 0, 0))  # mm -> m

        pos, quat = pose_list[i]
        T_base_to_end = pose_to_matrix(pos, quat)
        T_base_to_cam = T_base_to_end @ end_T_cam
        pcd.transform(T_base_to_cam)

        points = np.asarray(pcd.points)
        (cx, cy, cz, r), residuals = fit_sphere(points)

        print("球心坐标: ({:.2f}, {:.2f}, {:.2f}) mm".format(cx * 1000, cy * 1000, cz * 1000))
        print("球的直径: {:.2f} mm".format(2 * r * 1000))
        print_fitting_report(residuals * 1000)
        centers_mm.append([cx * 1000, cy * 1000, cz * 1000])

        # === 误差可视化上色（绿→红）===
        abs_residuals = np.abs(residuals)
        min_err, max_err = np.min(abs_residuals), np.max(abs_residuals)
        norm_err = (abs_residuals - min_err) / (max_err - min_err + 1e-8)
        colors = np.zeros((len(norm_err), 3))
        colors[:, 0] = norm_err
        colors[:, 1] = 1 - norm_err
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # === 球面显示 ===
        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
        x = cx + r * np.cos(u) * np.sin(v)
        y = cy + r * np.sin(u) * np.sin(v)
        z = cz + r * np.cos(v)
        sphere_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        sphere_pcd = o3d.geometry.PointCloud()
        sphere_pcd.points = o3d.utility.Vector3dVector(sphere_points)
        sphere_pcd.paint_uniform_color([1, 0, 0])  # 红色球面

        visual_geoms += [pcd, sphere_pcd]

    # === 球心轨迹可视化 ===
    centers_mm = np.array(centers_mm)
    trajectory_points = o3d.geometry.PointCloud()
    trajectory_points.points = o3d.utility.Vector3dVector(centers_mm / 1000.0)
    trajectory_points.paint_uniform_color([0, 0, 1])  # 蓝色点

    lines = [[i, i + 1] for i in range(len(centers_mm) - 1)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(centers_mm / 1000.0),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in lines])
    visual_geoms += [trajectory_points, line_set]

    # === 球心漂移误差报告 ===
    print("\n📊 球心漂移误差报告（单位 mm）")
    
    # 自动选择参考帧（误差最小 或 中位帧）
    center_array = np.array(centers_mm)
    mean_center = np.mean(center_array, axis=0)
    distances_to_mean = np.linalg.norm(center_array - mean_center, axis=1)
    best_idx = np.argmin(distances_to_mean)  # 最小误差帧
    ref_center = center_array[best_idx]
    print("自动选定第 {} 帧作为参考帧（球心最接近均值）".format(best_idx))
    diffs = center_array - ref_center
    
    distances = np.linalg.norm(diffs, axis=1)
    print("参考球心: ({:.2f}, {:.2f}, {:.2f}) mm".format(*ref_center))
    for i, (d, dist) in enumerate(zip(diffs, distances)):
        if i == best_idx: continue  # 排除参考帧
        print("帧 {} 偏移 dx={:.2f}, dy={:.2f}, dz={:.2f}, 总偏移={:.2f} mm".format(
            i, d[0], d[1], d[2], dist))
    print("\n统计:")
    print("最大偏移: {:.2f} mm".format(np.max(np.delete(distances, best_idx))))
    print("最小偏移: {:.2f} mm".format(np.min(np.delete(distances, best_idx))))
    print("平均偏移: {:.2f} mm".format(np.mean(np.delete(distances, best_idx))))
    print("标准差  : {:.2f} mm".format(np.std(np.delete(distances, best_idx))))

    # === 可视化 ===
    o3d.visualization.draw_geometries(visual_geoms)

if __name__ == "__main__":
    main()
