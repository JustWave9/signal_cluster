import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import scipy.io
# ===== 标签对齐（关键）=====
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score,adjusted_rand_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from extract_feature import extract_feature  # 假设你的18维特征函数保存为extract_feature.py
from feature import ex_feature
from scipy.spatial.distance import cdist
import re

# =========================
# 文件名解析
# 命名规则：
# signal3_1_3_23001.000000_31000.000000_3000000.000000.mat
#        | | |
#        | | └── 第2个下划线后：hop时刻
#        | └──── 第1个下划线后：设备号
#        └────── signal后面的数字只是样本序号
# =========================
def parse_filename(filename: str):
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    # 期望至少:
    # ['signal3', '1', '3', '23001.000000', '31000.000000', '3000000.000000']
    if len(parts) < 3:
        raise ValueError(f"文件名格式不符合预期: {filename}")

    try:
        device_id = int(parts[1])   # 第一个下划线后
        hop_time = int(parts[2])    # 第二个下划线后
    except Exception as e:
        raise ValueError(f"无法解析设备号/跳频时刻: {filename}") from e

    return device_id, hop_time


def get_true_label(filename: str) -> int:
    device_id, _ = parse_filename(filename)
    return device_id


def get_hop_time(filename: str) -> int:
    _, hop_time = parse_filename(filename)
    return hop_time

def align_cluster_labels(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in y_pred])

# =========================
# 加入“同一时刻不能分到同一类”的约束
# 思路：
# 1) 用 KMeans 得到 centers
# 2) 对每个 hop_time 分组
# 3) 在该组内，用样本到各类中心的距离矩阵做匈牙利匹配
#    保证该时刻内一个类别只分配一个样本
# =========================
def constrained_assign_by_hop_time(X_scaled, file_list, kmeans, num_classes):
    centers = kmeans.cluster_centers_
    hop_times = np.array([get_hop_time(f) for f in file_list])

    # 先给一个默认值
    constrained_labels = -np.ones(len(file_list), dtype=int)

    unique_hops = np.unique(hop_times)

    for hop in unique_hops:
        idx = np.where(hop_times == hop)[0]
        X_group = X_scaled[idx]

        # 距离矩阵：组内样本 × 所有类别中心
        cost = cdist(X_group, centers, metric='euclidean')

        # 如果该时刻样本数 <= 类别数，可以直接做唯一分配
        if len(idx) <= num_classes:
            row_ind, col_ind = linear_sum_assignment(cost)
            # row_ind 是组内样本索引，col_ind 是分配到的类
            for r, c in zip(row_ind, col_ind):
                constrained_labels[idx[r]] = c
        else:
            # 理论上如果“同一时刻样本数 > 发射器数”，约束不可能完全满足
            # 先给最优唯一分配，剩余样本再按最近中心分配
            row_ind, col_ind = linear_sum_assignment(cost[:, :num_classes])

            assigned_rows = set(row_ind.tolist())
            for r, c in zip(row_ind, col_ind):
                constrained_labels[idx[r]] = c

            # 剩余样本只能退化为最近中心
            for local_r in range(len(idx)):
                if local_r not in assigned_rows:
                    constrained_labels[idx[local_r]] = np.argmin(cost[local_r])

    # 若有极少数没被分到（理论上不应发生），退回最近中心
    unassigned = np.where(constrained_labels < 0)[0]
    if len(unassigned) > 0:
        fallback_dist = cdist(X_scaled[unassigned], centers, metric='euclidean')
        constrained_labels[unassigned] = np.argmin(fallback_dist, axis=1)

    return constrained_labels

# 主程序参数
# =========================
# 读取 / 提取特征
# =========================

folder = r'D:\matrixlab\match_tar\test37rNBP_snr0_bb1'  # 修改为你的信号路径
Fs = 1e7  #采样率
save_path = "test8w5.npz"  # 保存标准化特征文件名

if os.path.exists(save_path):
    print("检测到已保存特征，直接加载...")
    data = np.load(save_path, allow_pickle=True)
    X_scaled = data["X_scaled"]
    file_list = data["file_list"].tolist()
else:
    print("未检测到保存文件，开始提取特征...")
    signal_list = []
    file_list = []
    features_list = []
    for file in tqdm(os.listdir(folder)):
        if file.endswith('.mat'):
            path = os.path.join(folder, file)
            data = scipy.io.loadmat(path)
            if 'x_bb1' in data:
                signal = data['x_bb1']
                if signal.ndim > 1:
                    signal = signal.flatten()
                # signal_list.append(signal)
                file_list.append(file)
                feature_vector = ex_feature(signal[np.newaxis, :],Fs)[0]
                features_list.append(feature_vector)
    # original_signal_matrix = np.vstack(signal_list)
    # feature_matrix=ex_feature(original_signal_matrix, Fs)
    # X = np.array(feature_matrix)
    X = np.array(features_list)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # # 保存标准化特征和文件名列表
    # np.savez(save_path, X_scaled=X_scaled, file_list=file_list)
    print(f"特征提取完毕，保存至 {save_path}")
# =========================
# 打印样本和特征
# =========================
for i in range(len(file_list)):
    print(file_list[i])
    print(X[i])

# =========================
# 构造真实标签
# 设备号从1开始，这里减1变成 0~num_classes-1
# =========================
y_true = np.array([get_true_label(fname) - 1 for fname in file_list],dtype=int)
num_classes = len(np.unique(y_true))

# KMeans 五类聚类
kmeans = KMeans(n_clusters=num_classes, random_state=3, n_init=20)
labels_kmeans = kmeans.fit_predict(X_scaled)
ari_raw= adjusted_rand_score(y_true, labels_kmeans)
print("原始 Kmeans ARI =", ari_raw)
# =========================
# 再加“同一时刻互斥”约束
# =========================
labels_constrained = constrained_assign_by_hop_time(
    X_scaled=X_scaled,
    file_list=file_list,
    kmeans=kmeans,
    num_classes=num_classes
)
ari_constrained = adjusted_rand_score(y_true, labels_constrained)
print("加入同一时刻互斥约束后的 ARI =", ari_constrained)


labels_aligned = align_cluster_labels(y_true, labels_constrained)

# ===== PCA 可视化 =====
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
palette = sns.color_palette("bright", num_classes)

for cls in range(num_classes):
    idx = y_true == cls
    plt.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        s=50,
        alpha=0.7,
        color=palette[cls],
        label=f"True Class {cls}"
    )


plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Visualization (5 Classes)")
plt.legend()
plt.grid(True)
plt.show()

# ===== 混淆矩阵 =====
cm = confusion_matrix(y_true, labels_aligned)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[f"Pred {i}" for i in range(num_classes)],
    yticklabels=[f"True {i}" for i in range(num_classes)]
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("KMeans Confusion Matrix (5 Classes)")
plt.tight_layout()
plt.show()

# ===== 分类报告 =====
print(classification_report(
    y_true,
    labels_aligned,
    target_names=[f"Class {i}" for i in range(num_classes)]
))


# # KMeans聚类
# kmeans = KMeans(n_clusters=5, random_state=3)
# labels = kmeans.fit_predict(X_scaled)
#
# # 输出聚类结果
# print("文件名 → 聚类类别")
# for fname, label in zip(file_list, labels):
#     print(f"{fname} → 类别 {label}")
#
# # 构造真实标签
# y_true = [get_true_label(fname)-1 for fname in file_list]
# acc1 = accuracy_score(y_true, labels)
# acc2 = accuracy_score(y_true, 1 - labels)
# if acc2 > acc1:
#     labels = 1 - labels  # 标签对齐
#
# # 根据真实标签绘制散点图
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
# plt.figure(figsize=(8,6))
# palette = sns.color_palette("bright", 2)
#
# for label in np.unique(y_true):
#     idx = np.array(y_true) == label
#     plt.scatter(X_pca[idx, 0], X_pca[idx, 1],
#                 label=f'True Class {label}', alpha=0.7, s=50, c=[palette[label]])
#
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.title(" PCA ")
# plt.legend()
# plt.grid(True)
# plt.show()
#
#
# # 混淆矩阵
# cm = confusion_matrix(y_true, labels)
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("KMeans Confusion Matrix")
# plt.tight_layout()
# plt.show()
#
# # 分类报告
# print(classification_report(y_true, labels, target_names=["Class 0", "Class 1"]))