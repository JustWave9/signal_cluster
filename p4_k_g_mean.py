import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    adjusted_rand_score,
)
import pandas as pd

# =========================
# 你的特征函数
# 请确保 feature.py 和本脚本在同一目录，或已加入 Python 路径
# =========================
from feature import ex_feature


# =========================
# 文件名解析
# 文件名示例：
# signal1_1_1_1.000000_10000.000000_3000000.000000.mat
# signal2_1_2_11001.000000_21000.000000_3500000.000000.mat
#
# 规则：
# 第1个下划线后 parts[1] -> 用户/设备号
# 第2个下划线后 parts[2] -> hop时刻
# =========================
def parse_filename(filename: str):
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    if len(parts) < 3:
        raise ValueError(f"文件名格式不符合预期: {filename}")

    try:
        device_id = int(parts[1])
        hop_time = int(parts[2])
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
# 聚类标签编号本身无语义，需要匈牙利匹配后再算 accuracy
# =========================
def align_cluster_labels(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in y_pred])


# =========================
# KMeans + 同一时刻互斥约束
# 思路：
# 对同一 hop_time 的样本，尽量分配到不同类中心
# =========================
def constrained_assign_by_hop_time_kmeans(X_scaled, file_list, kmeans, num_classes):
    centers = kmeans.cluster_centers_
    hop_times = np.array([get_hop_time(f) for f in file_list])

    constrained_labels = -np.ones(len(file_list), dtype=int)

    for hop in np.unique(hop_times):
        idx = np.where(hop_times == hop)[0]
        X_group = X_scaled[idx]
        cost = cdist(X_group, centers, metric='euclidean')

        if len(idx) <= num_classes:
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                constrained_labels[idx[r]] = c
        else:
            row_ind, col_ind = linear_sum_assignment(cost[:, :num_classes])
            assigned_rows = set(row_ind.tolist())

            for r, c in zip(row_ind, col_ind):
                constrained_labels[idx[r]] = c

            for local_r in range(len(idx)):
                if local_r not in assigned_rows:
                    constrained_labels[idx[local_r]] = np.argmin(cost[local_r])

    # 回退处理
    unassigned = np.where(constrained_labels < 0)[0]
    if len(unassigned) > 0:
        fallback_dist = cdist(X_scaled[unassigned], centers, metric='euclidean')
        constrained_labels[unassigned] = np.argmin(fallback_dist, axis=1)

    return constrained_labels


# =========================
# GMM + 同一时刻互斥约束
# 用后验概率做代价：
# cost(i,k) = -log p(class=k | x_i)
# =========================
def constrained_assign_by_hop_time_gmm(X_scaled, file_list, gmm, num_classes, eps=1e-12):
    hop_times = np.array([get_hop_time(f) for f in file_list])

    prob = gmm.predict_proba(X_scaled)
    prob = np.clip(prob, eps, 1.0)

    constrained_labels = -np.ones(len(file_list), dtype=int)

    for hop in np.unique(hop_times):
        idx = np.where(hop_times == hop)[0]
        cost = -np.log(prob[idx])

        n_group = len(idx)

        if n_group <= num_classes:
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                constrained_labels[idx[r]] = c
        else:
            row_ind, col_ind = linear_sum_assignment(cost[:, :num_classes])
            assigned_rows = set(row_ind.tolist())

            for r, c in zip(row_ind, col_ind):
                constrained_labels[idx[r]] = c

            for local_r in range(n_group):
                if local_r not in assigned_rows:
                    constrained_labels[idx[local_r]] = np.argmax(prob[idx[local_r]])

    # 回退处理
    unassigned = np.where(constrained_labels < 0)[0]
    if len(unassigned) > 0:
        constrained_labels[unassigned] = np.argmax(prob[unassigned], axis=1)

    return constrained_labels


# =========================
# 读取或提取特征（支持缓存）
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

        feat = ex_feature(signal[np.newaxis, :], Fs)[0]
        features_list.append(feat)
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
# 单个文件夹评估
# 返回：KMeans/GMM 约束前后 accuracy / ARI
# =========================
def evaluate_one_folder(folder, Fs=1e7, cache_dir=None):
    X_scaled, file_list, y_true = load_or_extract_features(folder, Fs=Fs, cache_dir=cache_dir)

    num_classes = len(np.unique(y_true))
    if num_classes < 2:
        raise ValueError(f"{folder} 中类别数不足，无法聚类。")

    # ===== KMeans：约束前 =====
    kmeans = KMeans(n_clusters=num_classes, random_state=3, n_init=20)
    labels_kmeans_before = kmeans.fit_predict(X_scaled)

    ari_kmeans_before = adjusted_rand_score(y_true, labels_kmeans_before)
    labels_kmeans_before_aligned = align_cluster_labels(y_true, labels_kmeans_before)
    acc_kmeans_before = accuracy_score(y_true, labels_kmeans_before_aligned)

    # ===== KMeans：约束后 =====
    labels_kmeans_after = constrained_assign_by_hop_time_kmeans(
        X_scaled=X_scaled,
        file_list=file_list,
        kmeans=kmeans,
        num_classes=num_classes
    )
    ari_kmeans_after = adjusted_rand_score(y_true, labels_kmeans_after)
    labels_kmeans_after_aligned = align_cluster_labels(y_true, labels_kmeans_after)
    acc_kmeans_after = accuracy_score(y_true, labels_kmeans_after_aligned)

    # ===== GMM：约束前 =====
    gmm = GaussianMixture(
        n_components=num_classes,
        covariance_type='full',
        random_state=3,
        n_init=10
    )
    gmm.fit(X_scaled)
    labels_gmm_before = gmm.predict(X_scaled)

    ari_gmm_before = adjusted_rand_score(y_true, labels_gmm_before)
    labels_gmm_before_aligned = align_cluster_labels(y_true, labels_gmm_before)
    acc_gmm_before = accuracy_score(y_true, labels_gmm_before_aligned)

    # ===== GMM：约束后 =====
    labels_gmm_after = constrained_assign_by_hop_time_gmm(
        X_scaled=X_scaled,
        file_list=file_list,
        gmm=gmm,
        num_classes=num_classes
    )
    ari_gmm_after = adjusted_rand_score(y_true, labels_gmm_after)
    labels_gmm_after_aligned = align_cluster_labels(y_true, labels_gmm_after)
    acc_gmm_after = accuracy_score(y_true, labels_gmm_after_aligned)

    return {
        "num_samples": len(file_list),

        "acc_kmeans_before": acc_kmeans_before,
        "acc_kmeans_after": acc_kmeans_after,
        "ari_kmeans_before": ari_kmeans_before,
        "ari_kmeans_after": ari_kmeans_after,

        "acc_gmm_before": acc_gmm_before,
        "acc_gmm_after": acc_gmm_after,
        "ari_gmm_before": ari_gmm_before,
        "ari_gmm_after": ari_gmm_after,
    }


# =========================
# 主程序
# =========================
def main():
    # ========= 这里改成你的总目录 =========
    base_dir = r'D:\matrixlab\match_tar'

    # ========= 你的 SNR 列表 =========
    snr_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

    # ========= 每个 SNR 的重复实验次数 =========
    run_list = [1, 2, 3, 4, 5]

    Fs = 1e7

    # 全局缓存目录
    global_cache_dir = os.path.join(base_dir, "_feature_cache")
    os.makedirs(global_cache_dir, exist_ok=True)

    results = []

    print("\n========== 开始按 SNR × 5次重复实验评估 KMeans / GMM ==========\n")

    for snr in snr_list:
        acc_kmeans_before_runs = []
        acc_kmeans_after_runs = []
        ari_kmeans_before_runs = []
        ari_kmeans_after_runs = []

        acc_gmm_before_runs = []
        acc_gmm_after_runs = []
        ari_gmm_before_runs = []
        ari_gmm_after_runs = []

        num_samples_runs = []

        for run_idx in run_list:
            folder = os.path.join(base_dir, f"test37rNBP_snr{snr}_run{run_idx}_bb1")

            try:
                metrics = evaluate_one_folder(folder, Fs=Fs, cache_dir=global_cache_dir)

                acc_kmeans_before_runs.append(metrics["acc_kmeans_before"])
                acc_kmeans_after_runs.append(metrics["acc_kmeans_after"])
                ari_kmeans_before_runs.append(metrics["ari_kmeans_before"])
                ari_kmeans_after_runs.append(metrics["ari_kmeans_after"])

                acc_gmm_before_runs.append(metrics["acc_gmm_before"])
                acc_gmm_after_runs.append(metrics["acc_gmm_after"])
                ari_gmm_before_runs.append(metrics["ari_gmm_before"])
                ari_gmm_after_runs.append(metrics["ari_gmm_after"])

                num_samples_runs.append(metrics["num_samples"])

                print(
                    f"SNR={snr:>2} dB, run={run_idx} | "
                    f"KMeans Acc(b/a)={metrics['acc_kmeans_before']:.4f}/{metrics['acc_kmeans_after']:.4f} | "
                    f"KMeans ARI(b/a)={metrics['ari_kmeans_before']:.4f}/{metrics['ari_kmeans_after']:.4f} | "
                    f"GMM Acc(b/a)={metrics['acc_gmm_before']:.4f}/{metrics['acc_gmm_after']:.4f} | "
                    f"GMM ARI(b/a)={metrics['ari_gmm_before']:.4f}/{metrics['ari_gmm_after']:.4f}"
                )

            except Exception as e:
                print(f"[失败] SNR={snr} dB, run={run_idx} -> {e}")

        if len(acc_kmeans_before_runs) > 0:
            results.append({
                "snr": snr,
                "num_valid_runs": len(acc_kmeans_before_runs),
                "num_samples_mean": np.mean(num_samples_runs),

                "acc_kmeans_before_mean": np.mean(acc_kmeans_before_runs),
                "acc_kmeans_before_std": np.std(acc_kmeans_before_runs, ddof=1) if len(acc_kmeans_before_runs) > 1 else 0.0,
                "acc_kmeans_after_mean": np.mean(acc_kmeans_after_runs),
                "acc_kmeans_after_std": np.std(acc_kmeans_after_runs, ddof=1) if len(acc_kmeans_after_runs) > 1 else 0.0,

                "ari_kmeans_before_mean": np.mean(ari_kmeans_before_runs),
                "ari_kmeans_before_std": np.std(ari_kmeans_before_runs, ddof=1) if len(ari_kmeans_before_runs) > 1 else 0.0,
                "ari_kmeans_after_mean": np.mean(ari_kmeans_after_runs),
                "ari_kmeans_after_std": np.std(ari_kmeans_after_runs, ddof=1) if len(ari_kmeans_after_runs) > 1 else 0.0,

                "acc_gmm_before_mean": np.mean(acc_gmm_before_runs),
                "acc_gmm_before_std": np.std(acc_gmm_before_runs, ddof=1) if len(acc_gmm_before_runs) > 1 else 0.0,
                "acc_gmm_after_mean": np.mean(acc_gmm_after_runs),
                "acc_gmm_after_std": np.std(acc_gmm_after_runs, ddof=1) if len(acc_gmm_after_runs) > 1 else 0.0,

                "ari_gmm_before_mean": np.mean(ari_gmm_before_runs),
                "ari_gmm_before_std": np.std(ari_gmm_before_runs, ddof=1) if len(ari_gmm_before_runs) > 1 else 0.0,
                "ari_gmm_after_mean": np.mean(ari_gmm_after_runs),
                "ari_gmm_after_std": np.std(ari_gmm_after_runs, ddof=1) if len(ari_gmm_after_runs) > 1 else 0.0,
            })

    if len(results) == 0:
        raise RuntimeError("所有 SNR 都处理失败，请检查路径、变量名和文件夹命名。")

    results = sorted(results, key=lambda x: x["snr"])

    snrs = [r["snr"] for r in results]

    # =========================
    # 图1：平均 Classification Accuracy vs SNR
    # =========================
    plt.figure(figsize=(9, 5.5))

    plt.errorbar(
        snrs,
        [r["acc_kmeans_before_mean"] for r in results],
        yerr=[r["acc_kmeans_before_std"] for r in results],
        marker='s', linewidth=2, capsize=4,
        label='KMeans - before constraint'
    )

    plt.errorbar(
        snrs,
        [r["acc_kmeans_after_mean"] for r in results],
        yerr=[r["acc_kmeans_after_std"] for r in results],
        marker='o', linewidth=2, capsize=4,
        label='KMeans - after constraint'
    )

    plt.errorbar(
        snrs,
        [r["acc_gmm_before_mean"] for r in results],
        yerr=[r["acc_gmm_before_std"] for r in results],
        marker='^', linewidth=2, capsize=4,
        label='GMM - before constraint'
    )

    plt.errorbar(
        snrs,
        [r["acc_gmm_after_mean"] for r in results],
        yerr=[r["acc_gmm_after_std"] for r in results],
        marker='d', linewidth=2, capsize=4,
        label='GMM - after constraint'
    )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Classification Accuracy")
    plt.title("Average Classification Accuracy vs SNR")
    plt.grid(True)
    plt.xticks(snrs)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig("avg_classification_accuracy_vs_snr_kmeans_gmm.png", dpi=300)
    plt.show()

    # =========================
    # 图2：平均 ARI vs SNR
    # =========================
    plt.figure(figsize=(9, 5.5))

    plt.errorbar(
        snrs,
        [r["ari_kmeans_before_mean"] for r in results],
        yerr=[r["ari_kmeans_before_std"] for r in results],
        marker='s', linewidth=2, capsize=4,
        label='KMeans - before constraint'
    )

    plt.errorbar(
        snrs,
        [r["ari_kmeans_after_mean"] for r in results],
        yerr=[r["ari_kmeans_after_std"] for r in results],
        marker='o', linewidth=2, capsize=4,
        label='KMeans - after constraint'
    )

    plt.errorbar(
        snrs,
        [r["ari_gmm_before_mean"] for r in results],
        yerr=[r["ari_gmm_before_std"] for r in results],
        marker='^', linewidth=2, capsize=4,
        label='GMM - before constraint'
    )

    plt.errorbar(
        snrs,
        [r["ari_gmm_after_mean"] for r in results],
        yerr=[r["ari_gmm_after_std"] for r in results],
        marker='d', linewidth=2, capsize=4,
        label='GMM - after constraint'
    )

    plt.xlabel("SNR (dB)")
    plt.ylabel("ARI")
    plt.title("Average ARI vs SNR")
    plt.grid(True)
    plt.xticks(snrs)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig("avg_ari_vs_snr_kmeans_gmm.png", dpi=300)
    plt.show()

    # =========================
    # 保存结果表
    # =========================
    df = pd.DataFrame(results)
    df.to_csv("snr_constraint_comparison_kmeans_gmm_avg5.csv", index=False, encoding="utf-8-sig")

    print("\n========== 评估完成 ==========")
    print(df)
    print("\n已保存:")
    print(" - avg_classification_accuracy_vs_snr_kmeans_gmm.png")
    print(" - avg_ari_vs_snr_kmeans_gmm.png")
    print(" - snr_constraint_comparison_kmeans_gmm_avg5.csv")


if __name__ == "__main__":
    main()