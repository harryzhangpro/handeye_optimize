# handeye_optimize

球心点云工具(请缺啥装啥)

#### 球心点云拼接脚本
进入mapping.py修改路径  
```
# ================== 配置路径 ==================
base_dir = "./0411"  # 点云文件夹
record_file = "aubo_record_2025-04-11_22-15-34.txt" # 机械臂数据文件
calib_file = "cal.txt" # 手眼转换矩阵
```

#### 球心点云质量检测
```
python spheres.py --base_dir xxx(点云所在文件夹) \
                  --robot_file xxx.txt(机械臂数据文件) \
                  --calib_file xxx.txt(手眼转换矩阵)
```
输出：球心漂移误差报告、拟合误差报告、拟合球心坐标、拟合球的直径

#### 手眼标定矩阵优化
```
python optimize.py --base_dir xxx(点云所在文件夹) \
                   --robot_file xxx.txt(机械臂数据文件) \
                   --calib_file xxx.txt(手眼转换矩阵) \
                   --downsample xxx(float, 点云降采样参数) \
                   --Epoch xx(int, 迭代次数)
```
输出：优化后的矩阵optimized.txt，可以喂给上面两个.py使用
