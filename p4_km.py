import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    adjusted_rand_score,
)
import pandas as pd

# 你的特征函数
from feature import ex_feature


# =========================
# 文件名解析
# 命名规则示例：
# signal3_1_3_23001.000000_31000.000000_3000000.000000.mat
#        | | |
#        | | └── 第2个下划线后：hop时刻
#        | └──── 第1个下划线后：设备号
#        └────── signal后的数字只是样本序号
# =========================
def parse_filename(filename: str):
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
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


# =========================
# 标签对齐
# 聚类标签本身无序，先对齐再算 accuracy
# =========================
def align_cluster_labels(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in y_pred])


# =========================
# 同一时刻互斥约束
# 思路：
# 1) 用 KMeans 得到 centers
# 2) 按 hop_time 分组
# 3) 对每个 hop_time 组内，用样本到各类中心的距离矩阵做匈牙利匹配
#    保证同一时刻一个类别只分配一个样本
# =========================
def constrained_assign_by_hop_time(X_scaled, file_list, kmeans, num_classes):
    centers = kmeans.cluster_centers_
    hop_times = np.array([get_hop_time(f) for f in file_list])

    constrained_labels = -np.ones(len(file_list), dtype=int)
    unique_hops = np.unique(hop_times)

    for hop in unique_hops:
        idx = np.where(hop_times == hop)[0]
        X_group = X_scaled[idx]

        # 距离矩阵：组内样本 × 所有类别中心
        cost = cdist(X_group, centers, metric='euclidean')

        # 如果该时刻样本数 <= 类别数，可唯一分配
        if len(idx) <= num_classes:
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                constrained_labels[idx[r]] = c
        else:
            # 若同一时刻样本数 > 类别数，则无法完全满足互斥
            # 先做最优唯一分配，剩余样本退化到最近中心
            row_ind, col_ind = linear_sum_assignment(cost[:, :num_classes])

            assigned_rows = set(row_ind.tolist())
            for r, c in zip(row_ind, col_ind):
                constrained_labels[idx[r]] = c

            for local_r in range(len(idx)):
                if local_r not in assigned_rows:
                    constrained_labels[idx[local_r]] = np.argmin(cost[local_r])

    # 极少数未分配样本，回退最近中心
    unassigned = np.where(constrained_labels < 0)[0]
    if len(unassigned) > 0:
        fallback_dist = cdist(X_scaled[unassigned], centers, metric='euclidean')
        constrained_labels[unassigned] = np.argmin(fallback_dist, axis=1)

    return constrained_labels


# =========================
# 从一个文件夹读取样本并提特征
# 支持缓存 npz，避免重复提特征
# =========================
def load_or_extract_features(folder, Fs=1e7, cache_dir=None):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"文件夹不存在: {folder}")

    folder_name = os.path.basename(folder)

    if cache_dir is None:
        cache_dir = os.path.join(folder, "_cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"{folder_name}_features.npz")

    if os.path.exists(cache_path):
        print(f"[缓存] 直接加载: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        X_scaled = data["X_scaled"]
        file_list = data["file_list"].tolist()
        y_true = data["y_true"]
        return X_scaled, file_list, y_true

    print(f"[提特征] 开始处理: {folder_name}")
    file_list = []
    features_list = []

    mat_files = [f for f in os.listdir(folder) if f.endswith('.mat')]
    mat_files.sort()

    for file in tqdm(mat_files, desc=f"提特征 {folder_name}"):
        path = os.path.join(folder, file)
        data = scipy.io.loadmat(path)

        if 'x_bb1' not in data:
            continue

        signal = data['x_bb1']
        if signal.ndim > 1:
            signal = signal.flatten()

        feature_vector = ex_feature(signal[np.newaxis, :], Fs)[0]
        features_list.append(feature_vector)
        file_list.append(file)

    if len(file_list) == 0:
        raise ValueError(f"{folder} 中没有读到有效 .mat 文件或变量 x_bb1")

    X = np.array(features_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_true = np.array([get_true_label(fname) - 1 for fname in file_list], dtype=int)

    np.savez(
        cache_path,
        X_scaled=X_scaled,
        file_list=np.array(file_list, dtype=object),
        y_true=y_true
    )
    print(f"[缓存] 已保存: {cache_path}")

    return X_scaled, file_list, y_true


# =========================
# 评估一个 SNR 文件夹
# 输出：互斥约束前后的 accuracy / ARI
# =========================
def evaluate_one_folder(folder, Fs=1e7, cache_dir=None):
    X_scaled, file_list, y_true = load_or_extract_features(folder, Fs=Fs, cache_dir=cache_dir)

    num_classes = len(np.unique(y_true))
    if num_classes < 2:
        raise ValueError(f"{folder} 中类别数不足，无法聚类。")

    # ===== 约束前：原始 KMeans =====
    kmeans = KMeans(n_clusters=num_classes, random_state=3, n_init=20)
    labels_raw = kmeans.fit_predict(X_scaled)

    ari_before = adjusted_rand_score(y_true, labels_raw)

    labels_raw_aligned = align_cluster_labels(y_true, labels_raw)
    acc_before = accuracy_score(y_true, labels_raw_aligned)

    # ===== 约束后：同一时刻互斥 =====
    labels_constrained = constrained_assign_by_hop_time(
        X_scaled=X_scaled,
        file_list=file_list,
        kmeans=kmeans,
        num_classes=num_classes
    )

    ari_after = adjusted_rand_score(y_true, labels_constrained)

    labels_constrained_aligned = align_cluster_labels(y_true, labels_constrained)
    acc_after = accuracy_score(y_true, labels_constrained_aligned)

    return {
        "num_samples": len(file_list),
        "accuracy_before": acc_before,
        "accuracy_after": acc_after,
        "ari_before": ari_before,
        "ari_after": ari_after,
    }


# =========================
# 主程序
# =========================
def main():
    # ===== 这里改成你的总目录 =====
    base_dir = r'D:\matrixlab\match_tar'

    # 需要评估的 SNR
    snr_list = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,15,20]

    # 采样率
    Fs = 1e7

    # 缓存目录（可选）
    global_cache_dir = os.path.join(base_dir, "_feature_cache")
    os.makedirs(global_cache_dir, exist_ok=True)

    results = []

    print("\n========== 开始按 SNR 评估 ==========\n")

    for snr in snr_list:
        folder = os.path.join(base_dir, f"test37rNBP_snr{snr}_bb1")

        try:
            metrics = evaluate_one_folder(folder, Fs=Fs, cache_dir=global_cache_dir)
            metrics["snr"] = snr
            results.append(metrics)

            print(
                f"SNR={snr:>3} dB | "
                f"样本数={metrics['num_samples']:>4} | "
                f"Acc(before)={metrics['accuracy_before']:.4f} | "
                f"Acc(after)={metrics['accuracy_after']:.4f} | "
                f"ARI(before)={metrics['ari_before']:.4f} | "
                f"ARI(after)={metrics['ari_after']:.4f}"
            )

        except Exception as e:
            print(f"[失败] SNR={snr} dB -> {e}")

    if len(results) == 0:
        raise RuntimeError("所有 SNR 文件夹都处理失败，请检查路径、变量名和文件格式。")

    # 按 SNR 排序
    results = sorted(results, key=lambda x: x["snr"])

    snrs = [r["snr"] for r in results]
    acc_before = [r["accuracy_before"] for r in results]
    acc_after = [r["accuracy_after"] for r in results]
    ari_before = [r["ari_before"] for r in results]
    ari_after = [r["ari_after"] for r in results]

    # =========================
    # 图1：Classification Accuracy 前后对比
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(snrs, acc_before, marker='s', linewidth=2, label='Before mutual-exclusion constraint')
    plt.plot(snrs, acc_after, marker='o', linewidth=2, label='After mutual-exclusion constraint')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy vs SNR")
    plt.grid(True)
    plt.xticks(snrs)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig("classification_accuracy_vs_snr.png", dpi=300)
    plt.show()

    # =========================
    # 图2：ARI 前后对比
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(snrs, ari_before, marker='s', linewidth=2, label='Before mutual-exclusion constraint')
    plt.plot(snrs, ari_after, marker='o', linewidth=2, label='After mutual-exclusion constraint')
    plt.xlabel("SNR (dB)")
    plt.ylabel("ARI")
    plt.title("ARI vs SNR")
    plt.grid(True)
    plt.xticks(snrs)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ari_vs_snr.png", dpi=300)
    plt.show()

    # =========================
    # 保存结果表
    # =========================
    df = pd.DataFrame(results)
    df = df[[
        "snr",
        "num_samples",
        "accuracy_before",
        "accuracy_after",
        "ari_before",
        "ari_after"
    ]]
    df.to_csv("snr_constraint_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n========== 评估完成 ==========")
    print(df)
    print("\n已保存:")
    print(" - classification_accuracy_vs_snr.png")
    print(" - ari_vs_snr.png")
    print(" - snr_constraint_comparison.csv")


if __name__ == "__main__":
    main()