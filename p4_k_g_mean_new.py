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
from sklearn.metrics import confusion_matrix, accuracy_score, adjusted_rand_score
import pandas as pd

from feature import ex_feature

# =========================================================
# 文件名解析
# =========================================================
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


# =========================================================
# 标签对齐（仅用于 accuracy 评估）
# =========================================================
def align_cluster_labels(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping.get(label, label) for label in y_pred])


# =========================================================
# KMeans + 同一时刻互斥约束
# =========================================================
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

    unassigned = np.where(constrained_labels < 0)[0]
    if len(unassigned) > 0:
        fallback_dist = cdist(X_scaled[unassigned], centers, metric='euclidean')
        constrained_labels[unassigned] = np.argmin(fallback_dist, axis=1)

    return constrained_labels


# =========================================================
# GMM + 同一时刻互斥约束
# =========================================================
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

    unassigned = np.where(constrained_labels < 0)[0]
    if len(unassigned) > 0:
        constrained_labels[unassigned] = np.argmax(prob[unassigned], axis=1)

    return constrained_labels


# =========================================================
# 全部特征顺序（必须与 feature.py 的 ex_feature 输出严格一致）
# =========================================================
FEATURE_NAMES = [
    "rho",
    "c20_re_n",
    "c20_im_n",
    "dphi_std",
    "ph",
    "pn_b3",
    "y_envelope_mean",
    "P_o",
    "P_y",
    "P_k",
    "P_x",
    "R_HT",
    "J_HT",
    "SNRE",
    "Db",
    "Di",
    "LZC_y",
    "P_U",
    "P_O",
    "P_Y",
    "P_K",
    "P_X",
]
FEATURE_INDEX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}

# 三组特征组合
FEATURE_COMBOS = {
    "F1: P_u + u_a": ["P_U", "y_envelope_mean"],
    "F2: P_u + u_a + J": ["P_U", "y_envelope_mean", "J_HT"],
    "F3: P_u + u_a + J + D_b": ["P_U", "y_envelope_mean", "J_HT", "Db"],
}


# =========================================================
# 读取或提取原始 22 维特征（支持缓存）
# 缓存原始特征矩阵，不在这里做 StandardScaler
# =========================================================
def load_or_extract_feature_bank(folder, Fs=1e7, cache_dir=None):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"文件夹不存在: {folder}")

    folder_name = os.path.basename(folder)

    if cache_dir is None:
        cache_dir = os.path.join(folder, "_cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"{folder_name}_feature_bank_raw.npz")

    if os.path.exists(cache_path):
        print(f"[缓存] 直接加载: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        X_full = data["X_full"]
        file_list = data["file_list"].tolist()
        y_true = data["y_true"]
        return X_full, file_list, y_true

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

    X_full = np.array(features_list, dtype=float)
    y_true = np.array([get_true_label(fname) - 1 for fname in file_list], dtype=int)

    np.savez(
        cache_path,
        X_full=X_full,
        file_list=np.array(file_list, dtype=object),
        y_true=y_true,
    )
    print(f"[缓存] 已保存: {cache_path}")

    return X_full, file_list, y_true


# =========================================================
# 选择组合特征并标准化
# =========================================================
def select_and_scale_features(X_full, selected_feature_names):
    missing = [f for f in selected_feature_names if f not in FEATURE_INDEX]
    if missing:
        raise KeyError(f"未找到特征: {missing}")

    idx = [FEATURE_INDEX[f] for f in selected_feature_names]
    X_sel = X_full[:, idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)
    return X_scaled


# =========================================================
# 单个文件夹 + 单个特征组合评估
# 返回：KMeans/GMM 约束前后 accuracy / ARI
# =========================================================
def evaluate_one_folder_one_combo(folder, selected_feature_names, Fs=1e7, cache_dir=None):
    X_full, file_list, y_true = load_or_extract_feature_bank(folder, Fs=Fs, cache_dir=cache_dir)
    X_scaled = select_and_scale_features(X_full, selected_feature_names)

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
        num_classes=num_classes,
    )
    ari_kmeans_after = adjusted_rand_score(y_true, labels_kmeans_after)
    labels_kmeans_after_aligned = align_cluster_labels(y_true, labels_kmeans_after)
    acc_kmeans_after = accuracy_score(y_true, labels_kmeans_after_aligned)

    # ===== GMM：约束前 =====
    gmm = GaussianMixture(
        n_components=num_classes,
        covariance_type='full',
        random_state=3,
        n_init=10,
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
        num_classes=num_classes,
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


# =========================================================
# 主程序
# =========================================================
def main():
    # ========= 改成你的总目录 =========
    base_dir = r'D:\matrixlab\match_tar\p4_signal'

    # ========= 输出目录 =========
    out_dir = os.path.join(base_dir, 'compare_selected_feature_groups')
    os.makedirs(out_dir, exist_ok=True)

    # ========= 你的 SNR 列表 =========
    snr_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

    # ========= 每个 SNR 的重复实验次数 =========
    run_list = [1, 2, 3, 4, 5]

    Fs = 1e7

    # 全局缓存目录
    global_cache_dir = os.path.join(base_dir, "_feature_cache_raw22")
    os.makedirs(global_cache_dir, exist_ok=True)

    summary_rows = []
    run_rows = []

    print("\n========== 开始评估三组特征组合（KMeans / GMM，含时隙约束前后） ==========" )

    for combo_name, selected_features in FEATURE_COMBOS.items():
        print(f"\n===== 当前组合: {combo_name} -> {selected_features} =====")

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
                    metrics = evaluate_one_folder_one_combo(
                        folder=folder,
                        selected_feature_names=selected_features,
                        Fs=Fs,
                        cache_dir=global_cache_dir,
                    )

                    acc_kmeans_before_runs.append(metrics["acc_kmeans_before"])
                    acc_kmeans_after_runs.append(metrics["acc_kmeans_after"])
                    ari_kmeans_before_runs.append(metrics["ari_kmeans_before"])
                    ari_kmeans_after_runs.append(metrics["ari_kmeans_after"])
                    acc_gmm_before_runs.append(metrics["acc_gmm_before"])
                    acc_gmm_after_runs.append(metrics["acc_gmm_after"])
                    ari_gmm_before_runs.append(metrics["ari_gmm_before"])
                    ari_gmm_after_runs.append(metrics["ari_gmm_after"])
                    num_samples_runs.append(metrics["num_samples"])

                    run_rows.append({
                        "combo": combo_name,
                        "selected_features": ", ".join(selected_features),
                        "snr": snr,
                        "run": run_idx,
                        **metrics,
                    })

                    print(
                        f"SNR={snr:>2} dB, run={run_idx} | {combo_name} | "
                        f"KMeans Acc(b/a)={metrics['acc_kmeans_before']:.4f}/{metrics['acc_kmeans_after']:.4f} | "
                        f"KMeans ARI(b/a)={metrics['ari_kmeans_before']:.4f}/{metrics['ari_kmeans_after']:.4f} | "
                        f"GMM Acc(b/a)={metrics['acc_gmm_before']:.4f}/{metrics['acc_gmm_after']:.4f} | "
                        f"GMM ARI(b/a)={metrics['ari_gmm_before']:.4f}/{metrics['ari_gmm_after']:.4f}"
                    )
                except Exception as e:
                    print(f"[失败] combo={combo_name}, SNR={snr} dB, run={run_idx} -> {e}")

            if len(acc_kmeans_before_runs) == 0:
                continue

            summary_rows.append({
                "combo": combo_name,
                "selected_features": ", ".join(selected_features),
                "snr": snr,
                "num_valid_runs": len(acc_kmeans_before_runs),
                "num_samples_mean": float(np.mean(num_samples_runs)),
                "acc_kmeans_before_mean": float(np.mean(acc_kmeans_before_runs)),
                "acc_kmeans_after_mean": float(np.mean(acc_kmeans_after_runs)),
                "ari_kmeans_before_mean": float(np.mean(ari_kmeans_before_runs)),
                "ari_kmeans_after_mean": float(np.mean(ari_kmeans_after_runs)),
                "acc_gmm_before_mean": float(np.mean(acc_gmm_before_runs)),
                "acc_gmm_after_mean": float(np.mean(acc_gmm_after_runs)),
                "ari_gmm_before_mean": float(np.mean(ari_gmm_before_runs)),
                "ari_gmm_after_mean": float(np.mean(ari_gmm_after_runs)),
                "delta_acc_kmeans": float(np.mean(acc_kmeans_after_runs) - np.mean(acc_kmeans_before_runs)),
                "delta_ari_kmeans": float(np.mean(ari_kmeans_after_runs) - np.mean(ari_kmeans_before_runs)),
                "delta_acc_gmm": float(np.mean(acc_gmm_after_runs) - np.mean(acc_gmm_before_runs)),
                "delta_ari_gmm": float(np.mean(ari_gmm_after_runs) - np.mean(ari_gmm_before_runs)),
            })

    if len(summary_rows) == 0:
        raise RuntimeError("所有组合和所有 SNR 都处理失败，请检查路径、变量名和文件夹命名。")

    df_summary = pd.DataFrame(summary_rows).sort_values(["combo", "snr"]).reset_index(drop=True)
    df_runs = pd.DataFrame(run_rows).sort_values(["combo", "snr", "run"]).reset_index(drop=True)

    # 保存表格
    summary_csv = os.path.join(out_dir, "selected_feature_group_summary.csv")
    runs_csv = os.path.join(out_dir, "selected_feature_group_runs.csv")
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    df_runs.to_csv(runs_csv, index=False, encoding="utf-8-sig")

    # 画图样式（不再画方差 bar）
    combo_styles = {
        "F1: P_u + u_a": {"kmeans": ('o', '-'), "gmm": ('o', '--')},
        "F2: P_u + u_a + J": {"kmeans": ('s', '-'), "gmm": ('s', '--')},
        "F3: P_u + u_a + J + D_b": {"kmeans": ('^', '-'), "gmm": ('^', '--')},
    }

    # =========================================================
    # 图1：不同特征组合在时隙约束后 的 Accuracy vs SNR
    # =========================================================
    plt.figure(figsize=(10, 6))
    for combo_name in FEATURE_COMBOS.keys():
        sub = df_summary[df_summary["combo"] == combo_name].sort_values("snr")
        mk_km, ls_km = combo_styles[combo_name]["kmeans"]
        mk_gm, ls_gm = combo_styles[combo_name]["gmm"]

        plt.plot(
            sub["snr"],
            sub["acc_kmeans_after_mean"],
            marker=mk_km,
            linestyle=ls_km,
            linewidth=2,
            label=f"KMeans | {combo_name}",
        )
        plt.plot(
            sub["snr"],
            sub["acc_gmm_after_mean"],
            marker=mk_gm,
            linestyle=ls_gm,
            linewidth=2,
            label=f"GMM | {combo_name}",
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Classification Accuracy")
    plt.title("Accuracy vs SNR (after slot-time constraint)")
    plt.grid(True)
    plt.xticks(snr_list)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=9)
    plt.tight_layout()
    acc_plot_path = os.path.join(out_dir, "accuracy_vs_snr_after_constraint_selected_groups.svg")
    plt.savefig(acc_plot_path, format="svg", bbox_inches="tight")
    plt.show()

    # =========================================================
    # 图2：不同特征组合在时隙约束后 的 ARI vs SNR
    # =========================================================
    plt.figure(figsize=(10, 6))
    for combo_name in FEATURE_COMBOS.keys():
        sub = df_summary[df_summary["combo"] == combo_name].sort_values("snr")
        mk_km, ls_km = combo_styles[combo_name]["kmeans"]
        mk_gm, ls_gm = combo_styles[combo_name]["gmm"]

        plt.plot(
            sub["snr"],
            sub["ari_kmeans_after_mean"],
            marker=mk_km,
            linestyle=ls_km,
            linewidth=2,
            label=f"KMeans | {combo_name}",
        )
        plt.plot(
            sub["snr"],
            sub["ari_gmm_after_mean"],
            marker=mk_gm,
            linestyle=ls_gm,
            linewidth=2,
            label=f"GMM | {combo_name}",
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("ARI")
    plt.title("ARI vs SNR (after slot-time constraint)")
    plt.grid(True)
    plt.xticks(snr_list)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=9)
    plt.tight_layout()
    ari_plot_path = os.path.join(out_dir, "ari_vs_snr_after_constraint_selected_groups.svg")
    plt.savefig(ari_plot_path, format="svg", bbox_inches="tight")
    plt.show()

    # =========================================================
    # 图3：时隙约束前后带来的性能增益（单独一张图，两行子图）
    # 不画 bar，只画提升量曲线
    # =========================================================
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    for combo_name in FEATURE_COMBOS.keys():
        sub = df_summary[df_summary["combo"] == combo_name].sort_values("snr")
        mk_km, ls_km = combo_styles[combo_name]["kmeans"]
        mk_gm, ls_gm = combo_styles[combo_name]["gmm"]

        axes[0].plot(
            sub["snr"],
            sub["delta_acc_kmeans"],
            marker=mk_km,
            linestyle=ls_km,
            linewidth=2,
            label=f"KMeans | {combo_name}",
        )
        axes[0].plot(
            sub["snr"],
            sub["delta_acc_gmm"],
            marker=mk_gm,
            linestyle=ls_gm,
            linewidth=2,
            label=f"GMM | {combo_name}",
        )

        axes[1].plot(
            sub["snr"],
            sub["delta_ari_kmeans"],
            marker=mk_km,
            linestyle=ls_km,
            linewidth=2,
            label=f"KMeans | {combo_name}",
        )
        axes[1].plot(
            sub["snr"],
            sub["delta_ari_gmm"],
            marker=mk_gm,
            linestyle=ls_gm,
            linewidth=2,
            label=f"GMM | {combo_name}",
        )

    axes[0].axhline(0, color='black', linewidth=1)
    axes[0].set_ylabel("ΔAccuracy")
    axes[0].set_title("Improvement brought by slot-time constraint")
    axes[0].grid(True)
    axes[0].legend(fontsize=9)

    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("ΔARI")
    axes[1].grid(True)
    axes[1].set_xticks(snr_list)

    plt.tight_layout()
    gain_plot_path = os.path.join(out_dir, "slot_time_constraint_gain_selected_groups.svg")
    plt.savefig(gain_plot_path, format="svg", bbox_inches="tight")
    plt.show()

    print("\n========== 评估完成 ==========")
    print(df_summary)
    print("\n已保存:")
    print(f" - {summary_csv}")
    print(f" - {runs_csv}")
    print(f" - {acc_plot_path}")
    print(f" - {ari_plot_path}")
    print(f" - {gain_plot_path}")


if __name__ == "__main__":
    main()
