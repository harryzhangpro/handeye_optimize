import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import open3d as o3d
import os, psutil
import argparse
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def load_pose_list(txt_path):
    pose_list = []
    with open(txt_path, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if "Position" in lines[i]:
                pos = eval(lines[i].split(":")[1])
                ori = eval(lines[i+1].split(":")[1])  # xyzw
                pose_list.append((np.array(pos), np.array(ori)))
    return pose_list

def pose_to_matrix(position, quat_xyzw, device):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = position
    return torch.tensor(T, dtype=torch.float32, device=device)

def load_pointclouds(pcd_dir, downsample, device):
    pcs = []
    for i in range(8):
        pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, f"point_cloud_{i:05d}.pcd"))
        pcd = pcd.voxel_down_sample(voxel_size=downsample)
        pcd.scale(0.001, center=(0,0,0))  # mm -> m
        pcs.append(torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device=device))
    return pcs

def fit_sphere_least_squares(points):
    A = torch.cat([-2 * points, torch.ones(points.shape[0], 1, device=points.device)], dim=1)
    b = -torch.sum(points**2, dim=1, keepdim=True)
    x = torch.linalg.lstsq(A, b).solution
    center = x[:3].squeeze()

    # 计算所有点到中心的距离，剔除异常点再拟合一次
    dists = torch.norm(points - center[None, :], dim=1)
    median = torch.median(dists)
    mad = torch.median(torch.abs(dists - median)) + 1e-6  # 避免除以 0
    threshold = median + 2.5 * mad
    inliers = points[dists < threshold]

    if inliers.shape[0] < 10:
        return center  # 点数太少不再拟合

    A_in = torch.cat([-2 * inliers, torch.ones(inliers.shape[0], 1, device=points.device)], dim=1)
    b_in = -torch.sum(inliers**2, dim=1, keepdim=True)
    x_refined = torch.linalg.lstsq(A_in, b_in).solution
    center_refined = x_refined[:3].squeeze()
    return center_refined

class HandeyeOptimizer(nn.Module):
    def __init__(self, init_matrix):
        super().__init__()
        init_rot = R.from_matrix(init_matrix[:3,:3]).as_rotvec()
        self.rotvec = nn.Parameter(torch.tensor(init_rot, dtype=torch.float32))
        self.trans = nn.Parameter(torch.tensor(init_matrix[:3,3], dtype=torch.float32))

    def forward(self, pose_list, pointcloud_list, device):
        R_eye = self.rodrigues(self.rotvec, device)
        T_eye = torch.eye(4, device=device)
        T_eye[:3,:3] = R_eye
        T_eye[:3,3] = self.trans

        centers = []
        for (pos, quat), points in zip(pose_list, pointcloud_list):
            T_base = pose_to_matrix(pos, quat, device)
            T_cam = T_base @ T_eye

            points_h = torch.cat([points, torch.ones(points.shape[0], 1, device=device)], dim=1)
            transformed = (T_cam @ points_h.T).T[:, :3]

            center = fit_sphere_least_squares(transformed)
            centers.append(center)

        centers = torch.stack(centers)
        mean = centers.mean(0)
        errors = torch.norm(centers - mean, dim=1)
        return errors.mean(), centers

    def rodrigues(self, rotvec, device):
        theta = torch.norm(rotvec)
        if theta.item() < 1e-6:
            return torch.eye(3, device=device)
        r = rotvec / theta
        K = torch.tensor([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ], device=device)
        Rm = torch.eye(3, device=device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K
        return Rm

def main(robot_file=None, calib_file=None, base_dir=None, downsample=1.0, Epoch=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pose_list = load_pose_list(robot_file)
    pointclouds = load_pointclouds(base_dir, downsample, device)

    with open(calib_file, "r") as f:
        content = f.read()
        import re
        matrix_str = re.search(r"\[\[.*?\]\]", content, re.DOTALL).group()
        end_T_cam_ori = np.array(eval(matrix_str))

    model = HandeyeOptimizer(end_T_cam_ori).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    print("初始 handeye 平移向量 (m):", model.trans.detach().cpu().numpy())
    print("初始 handeye 旋转向量 (axis-angle):", model.rotvec.detach().cpu().numpy())
    print("\n开始优化...\n")

    process = psutil.Process(os.getpid())
    loss_list = []
    lr_list = []

    for epoch in range(Epoch):
        mem = process.memory_info().rss / 1024**2
        print(f"\n[Epoch {epoch:03d}] Memory usage: {mem:.2f} MB")

        loss, centers = model(pose_list, pointclouds, device)
        print(f"Frame Centers (mm): {[c.tolist() for c in centers.cpu() * 1000]}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        lr_list.append(optimizer.param_groups[0]['lr'])
        loss_list.append(loss.item())
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")

    print("\n优化完成。")
    print("最终 handeye 平移向量 (m):", model.trans.detach().cpu().numpy())
    print("最终 handeye 旋转向量 (axis-angle):", model.rotvec.detach().cpu().numpy())

    R_final = model.rodrigues(model.rotvec, device).detach().cpu().numpy()
    T_final = np.eye(4)
    T_final[:3, :3] = R_final
    T_final[:3, 3] = model.trans.detach().cpu().numpy()
    with open("optimized.txt", "w") as f:
        f.write("optimized_T = " + repr(T_final.tolist()))
    print("\n✅ 已保存优化后的 handeye 矩阵到 optimized.txt")

    # === 绘制 Loss 曲线 ===
    plt.figure()
    plt.plot(loss_list, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Optimization Loss Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig("loss_curve.png")

    # === 绘制学习率曲线 ===
    plt.figure()
    plt.plot(lr_list, label='Learning Rate', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.legend()
    plt.savefig("lr_curve.png")
    plt.show()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--downsample", type=float, default=1.0, help="体素降采样大小，单位为米，默认 0.02")
    parser.add_argument("--base_dir", type=str, default="./0411", help="点云文件夹路径")
    parser.add_argument("--robot_file", type=str, default="aubo_record_2025-04-11_22-15-34.txt", help="机械臂位姿文件")
    parser.add_argument("--calib_file", type=str, default="cal.txt", help="手眼标定文件")
    parser.add_argument("--Epoch", type=int, default=500, help="迭代次数")
    args = parser.parse_args()

    main(robot_file=args.robot_file,
         calib_file=args.calib_file,
         base_dir=args.base_dir,
         downsample=args.downsample,
         Epoch=args.Epoch)
