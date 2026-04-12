import os
import glob
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from feature_old import ex_feature


# =========================================================
# 1) 当前最终使用的 13 维特征名
#    与新版 ex_feature 输出顺序保持一致
# =========================================================
FEATURE_NAMES = [
    "SNRE",
    "rho",
    "c20_re_n",
    "c20_im_n",
    "dphi_std",
    "P_in",
    "P_out",
    "y_envelope_mean",
    "R_HT",
    "J_HT",
    "Db",
    "Di",
    "LZC_y",
]


# =========================================================
# 2) 通用：如果 csv 已存在且不强制重算，则直接读取
# =========================================================
def load_or_compute_table(csv_path, compute_func, force_recompute=False, **kwargs):
    if os.path.exists(csv_path) and (not force_recompute):
        print(f"[LOAD] Found existing file: {csv_path}")
        df = pd.read_csv(csv_path)
        return df
    else:
        print(f"[COMPUTE] File not found or force_recompute=True: {csv_path}")
        df = compute_func(**kwargs)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] Saved to: {csv_path}")
        return df


def load_or_build_feature_table(
        root_dir,
        feature_csv,
        force_recompute=False,
        in_band_hz=(2e3, 5e4),
        out_band_hz=(5e4, 8e4),
):
    """
    如果 feature_table.csv 已存在且不强制重算，则直接读取；
    否则重新从 .mat 文件提取特征。
    """
    if os.path.exists(feature_csv) and (not force_recompute):
        print(f"[LOAD] Found existing feature table: {feature_csv}")
        df = pd.read_csv(feature_csv)
        return df
    else:
        print(f"[COMPUTE] Building feature table from .mat files...")
        df = build_feature_table(
            root_dir,
            save_csv=feature_csv,
            in_band_hz=in_band_hz,
            out_band_hz=out_band_hz,
        )
        return df


# =========================================================
# 3) 读取 .mat 文件
# =========================================================
def read_one_mat(mat_path: str):
    data = loadmat(mat_path)

    x_bb1 = data["x_bb1"].squeeze()
    user_id = int(np.squeeze(data["user_id"]))
    hop_id = int(np.squeeze(data["hop_id"]))
    mc_id = int(np.squeeze(data["mc_id"]))
    snr = int(np.squeeze(data["snr"]))
    hopping = int(np.squeeze(data["hopping"]))
    fs = float(np.squeeze(data["fs"]))
    Rb = float(np.squeeze(data["Rb"]))
    sample_len = int(np.squeeze(data["sample_len"]))
    f_center = float(np.squeeze(data["f_center"]))

    return {
        "x_bb1": x_bb1,
        "user_id": user_id,
        "hop_id": hop_id,
        "mc_id": mc_id,
        "snr": snr,
        "hopping": hopping,
        "fs": fs,
        "Rb": Rb,
        "sample_len": sample_len,
        "f_center": f_center,
        "path": mat_path,
    }


# =========================================================
# 4) 单个样本提特征
#    新版 ex_feature 支持固定带内/带外积分范围
# =========================================================
def extract_one_feature_vector(
        x: np.ndarray,
        fs: float,
        in_band_hz=(2e3, 5e4),
        out_band_hz=(5e4, 8e4),
):
    x = np.asarray(x).squeeze()

    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape={x.shape}")

    # ex_feature 要求二维输入：(N_samples, signal_len)
    signal_matrix = np.expand_dims(x, axis=0)

    feat = ex_feature(
        signal_matrix,
        fs,
        in_band_hz=in_band_hz,
        out_band_hz=out_band_hz,
        force_nonnegative=False,   # 第三章评分阶段不建议强制转正
    )

    feat = np.asarray(feat).squeeze()

    if feat.ndim != 1:
        raise ValueError(f"feature output must be 1-D after squeeze, got shape={feat.shape}")

    if len(feat) != len(FEATURE_NAMES):
        raise ValueError(
            f"feature dimension mismatch: got {len(feat)}, expected {len(FEATURE_NAMES)}"
        )

    return feat


# =========================================================
# 5) 批量构建总特征表
# =========================================================
def build_feature_table(
        root_dir: str,
        save_csv: str = None,
        in_band_hz=(2e3, 5e4),
        out_band_hz=(5e4, 8e4),
):
    mat_files = glob.glob(os.path.join(root_dir, "**", "*.mat"), recursive=True)
    mat_files = sorted(mat_files)

    if len(mat_files) == 0:
        raise FileNotFoundError(f"No .mat files found under: {root_dir}")

    rows = []

    for i, mat_path in enumerate(mat_files, 1):
        meta = read_one_mat(mat_path)

        try:
            feat = extract_one_feature_vector(
                meta["x_bb1"],
                meta["fs"],
                in_band_hz=in_band_hz,
                out_band_hz=out_band_hz,
            )
        except Exception as e:
            warnings.warn(f"Feature extraction failed: {mat_path}\n{e}")
            continue

        row = {
            "path": meta["path"],
            "user_id": meta["user_id"],
            "hop_id": meta["hop_id"],
            "mc_id": meta["mc_id"],
            "snr": meta["snr"],
            "hopping": meta["hopping"],
            "fs": meta["fs"],
            "Rb": meta["Rb"],
            "sample_len": meta["sample_len"],
            "f_center": meta["f_center"],
        }

        for k, name in enumerate(FEATURE_NAMES):
            row[name] = float(feat[k])

        rows.append(row)

        if i % 100 == 0:
            print(f"Processed {i}/{len(mat_files)} files...")

    df = pd.DataFrame(rows)

    if save_csv is not None:
        df.to_csv(save_csv, index=False, encoding="utf-8-sig")
        print(f"Feature table saved to: {save_csv}")

    return df


# =========================================================
# 6) 计算稳定性：改成更稳健的“类内标准差 / 类内平均绝对值”
#    原来的 std / abs(mean) 对 c20_re_n, c20_im_n 不稳定
# =========================================================
def compute_cv_table(df: pd.DataFrame, feature_names):
    rows = []

    for feat in feature_names:
        for (snr, hopping, user_id), sub in df.groupby(["snr", "hopping", "user_id"]):
            x = sub[feat].values.astype(float)

            mean_val = np.mean(x)
            std_val = np.std(x, ddof=1) if len(x) > 1 else 0.0
            mean_abs_val = np.mean(np.abs(x))

            cv_like = std_val / (mean_abs_val + 1e-12)

            rows.append({
                "snr": snr,
                "hopping": hopping,
                "user_id": user_id,
                "feature": feat,
                "mean": mean_val,
                "std": std_val,
                "mean_abs": mean_abs_val,
                "cv": cv_like,
            })

    grouped = pd.DataFrame(rows)
    summary = grouped.groupby(["snr", "hopping", "feature"])["cv"].mean().reset_index()
    return summary


# =========================================================
# 7) 计算可分性：Fisher score
# =========================================================
def fisher_score_one_condition(df_cond: pd.DataFrame, feat: str):
    class_means = df_cond.groupby("user_id")[feat].mean()
    global_mean = df_cond[feat].mean()

    sb = ((class_means - global_mean) ** 2).sum()

    sw = 0.0
    for uid, sub in df_cond.groupby("user_id"):
        mu = sub[feat].mean()
        sw += ((sub[feat] - mu) ** 2).sum()

    return sb / (sw + 1e-12)


def compute_fisher_table(df: pd.DataFrame, feature_names):
    rows = []

    for feat in feature_names:
        for (snr, hopping), sub in df.groupby(["snr", "hopping"]):
            score = fisher_score_one_condition(sub, feat)
            rows.append({
                "snr": snr,
                "hopping": hopping,
                "feature": feat,
                "fisher": score
            })

    return pd.DataFrame(rows)


# =========================================================
# 8) 单特征分类准确率
# =========================================================
def compute_single_feature_acc(df: pd.DataFrame, feature_names, n_splits=5):
    rows = []

    for feat in feature_names:
        for (snr, hopping), sub in df.groupby(["snr", "hopping"]):
            X = sub[[feat]].values.astype(float)
            y = sub["user_id"].values

            unique_classes, counts = np.unique(y, return_counts=True)
            if len(unique_classes) < 2:
                continue
            if counts.min() < n_splits:
                continue

            clf = make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=3)
            )

            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

            rows.append({
                "snr": snr,
                "hopping": hopping,
                "feature": feat,
                "acc_mean": scores.mean(),
                "acc_std": scores.std()
            })

    return pd.DataFrame(rows)


# =========================================================
# 9) 综合汇总表
# =========================================================
def build_summary_table(cv_df, fisher_df, acc_df):
    cv_mean = cv_df.groupby("feature")["cv"].mean().rename("avg_cv")
    fisher_mean = fisher_df.groupby("feature")["fisher"].mean().rename("avg_fisher")
    acc_mean = acc_df.groupby("feature")["acc_mean"].mean().rename("avg_acc")

    summary = pd.concat([cv_mean, fisher_mean, acc_mean], axis=1).reset_index()

    # 缺失值兜底
    if summary["avg_cv"].notna().any():
        worst_cv = summary["avg_cv"].dropna().max()
    else:
        worst_cv = 1.0

    summary["avg_cv"] = summary["avg_cv"].fillna(worst_cv)
    summary["avg_fisher"] = summary["avg_fisher"].fillna(0.0)
    summary["avg_acc"] = summary["avg_acc"].fillna(0.0)

    def norm_pos(x):
        x = x.astype(float)
        if x.max() - x.min() < 1e-12:
            return np.ones_like(x)
        return (x - x.min()) / (x.max() - x.min())

    def norm_neg(x):
        x = x.astype(float)
        if x.max() - x.min() < 1e-12:
            return np.ones_like(x)
        return (x.max() - x) / (x.max() - x.min())

    summary["score_cv"] = norm_neg(summary["avg_cv"].values)
    summary["score_fisher"] = norm_pos(summary["avg_fisher"].values)
    summary["score_acc"] = norm_pos(summary["avg_acc"].values)

    summary["final_score"] = (
        0.4 * summary["score_cv"] +
        0.3 * summary["score_fisher"] +
        0.3 * summary["score_acc"]
    )

    summary = summary.sort_values("final_score", ascending=False).reset_index(drop=True)
    return summary


# =========================================================
# 10) 画图：横轴 hopping，曲线是不同 SNR
# =========================================================
def plot_metric_vs_hopping(metric_df, feature_name, metric_col, ylabel, save_path=None):
    sub = metric_df[metric_df["feature"] == feature_name].copy()
    sub = sub.sort_values(["snr", "hopping"])

    plt.figure(figsize=(7, 5))
    for snr, d in sub.groupby("snr"):
        plt.plot(d["hopping"], d[metric_col], marker="o", linewidth=2, label=f"SNR={snr} dB")

    plt.xlabel("Hopping rate (hop/s)")
    plt.ylabel(ylabel)
    plt.title(f"{feature_name}: {ylabel} vs Hopping Rate")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format="svg", bbox_inches="tight")

    plt.show()


# =========================================================
# 11) 网格图
# =========================================================
def plot_metric_grid(metric_df, features, metric_col, ylabel, save_path=None):
    n_feat = len(features)
    ncols = 4
    nrows = math.ceil(n_feat / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
    axes = np.array(axes).reshape(-1)

    valid_vals = metric_df[metric_col].replace([np.inf, -np.inf], np.nan).dropna()

    if len(valid_vals) == 0:
        ymin, ymax = 0, 1
    else:
        ymin = valid_vals.min()
        ymax = valid_vals.max()

        if ymax - ymin < 1e-12:
            pad = 0.1 * (abs(ymax) + 1e-12)
        else:
            pad = 0.05 * (ymax - ymin)

        ymin = ymin - pad
        ymax = ymax + pad

    for idx, feat in enumerate(features):
        ax = axes[idx]
        sub = metric_df[metric_df["feature"] == feat].copy()
        sub = sub.sort_values(["snr", "hopping"])

        for snr, d in sub.groupby("snr"):
            ax.plot(
                d["hopping"],
                d[metric_col],
                marker="o",
                linewidth=1.5,
                label=f"SNR={snr} dB"
            )

        ax.set_xlabel("Hopping rate (hop/s)")
        ax.set_ylabel(ylabel)
        ax.set_title(feat)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(ymin, ymax)

    for j in range(n_feat, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(4, len(labels)),
        frameon=False
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        plt.savefig(save_path, format="svg", bbox_inches="tight")

    plt.show()


def make_log_fisher_df(fisher_df: pd.DataFrame):
    fisher_log_df = fisher_df.copy()
    fisher_log_df["fisher_log"] = np.log10(1.0 + fisher_log_df["fisher"].clip(lower=0))
    return fisher_log_df


def plot_summary_bar(summary_df, save_path=None):
    summary_df = summary_df.sort_values("final_score", ascending=False)

    plt.figure(figsize=(12, 5))
    plt.bar(summary_df["feature"], summary_df["final_score"])
    plt.xlabel("Feature")
    plt.ylabel("Final Score")
    plt.title("Comprehensive Ranking of Features")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format="svg", bbox_inches="tight")

    plt.show()


# =========================================================
# 12) 主函数
# =========================================================
def main():
    root_dir = r"D:\matrixlab\exp_data_m5_5_8"
    out_dir = r"D:\matrixlab\exp_result_m5_5_8_old"
    os.makedirs(out_dir, exist_ok=True)

    # 固定相位增量谱积分区间（所有信号统一）
    in_band_hz = (2e3, 5e4)
    out_band_hz = (5e4, 8e4)

    # 分项控制是否强制重算
    force_feature = False
    force_cv = False
    force_fisher = False
    force_acc = False
    force_summary = False

    feature_csv = os.path.join(out_dir, "feature_table.csv")
    cv_csv = os.path.join(out_dir, "cv_table.csv")
    fisher_csv = os.path.join(out_dir, "fisher_table.csv")
    acc_csv = os.path.join(out_dir, "acc_table.csv")
    summary_csv = os.path.join(out_dir, "feature_summary.csv")

    # 1. 特征表
    df = load_or_build_feature_table(
        root_dir=root_dir,
        feature_csv=feature_csv,
        force_recompute=force_feature,
        in_band_hz=in_band_hz,
        out_band_hz=out_band_hz,
    )

    print("\nFeature table head:")
    print(df.head())
    print("\nShape:", df.shape)

    # 2. CV
    cv_df = load_or_compute_table(
        csv_path=cv_csv,
        compute_func=compute_cv_table,
        force_recompute=force_cv,
        df=df,
        feature_names=FEATURE_NAMES
    )

    # 3. Fisher
    fisher_df = load_or_compute_table(
        csv_path=fisher_csv,
        compute_func=compute_fisher_table,
        force_recompute=force_fisher,
        df=df,
        feature_names=FEATURE_NAMES
    )

    # 4. Accuracy
    acc_df = load_or_compute_table(
        csv_path=acc_csv,
        compute_func=compute_single_feature_acc,
        force_recompute=force_acc,
        df=df,
        feature_names=FEATURE_NAMES,
        n_splits=5
    )

    # 5. Summary
    summary_df = load_or_compute_table(
        csv_path=summary_csv,
        compute_func=build_summary_table,
        force_recompute=force_summary,
        cv_df=cv_df,
        fisher_df=fisher_df,
        acc_df=acc_df
    )

    print("\nFeature summary:")
    print(summary_df)

    # 6. 画所有特征图
    selected_plot_features = FEATURE_NAMES

    plot_metric_grid(
        cv_df,
        selected_plot_features,
        metric_col="cv",
        ylabel="CV",
        save_path=os.path.join(out_dir, "cv_all_features.svg")
    )

    plot_metric_grid(
        fisher_df,
        selected_plot_features,
        metric_col="fisher",
        ylabel="Fisher Score",
        save_path=os.path.join(out_dir, "fisher_all_features.svg")
    )

    fisher_log_df = make_log_fisher_df(fisher_df)

    plot_metric_grid(
        fisher_log_df,
        selected_plot_features,
        metric_col="fisher_log",
        ylabel=r"$\log_{10}(1+\mathrm{Fisher})$",
        save_path=os.path.join(out_dir, "fisher_log_all_features.svg")
    )

    plot_metric_grid(
        acc_df,
        selected_plot_features,
        metric_col="acc_mean",
        ylabel="Accuracy",
        save_path=os.path.join(out_dir, "acc_all_features.svg")
    )

    plot_summary_bar(
        summary_df,
        save_path=os.path.join(out_dir, "feature_final_score.svg")
    )

    print(f"\nAll results saved to: {out_dir}")


if __name__ == "__main__":
    main()