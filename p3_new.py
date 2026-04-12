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

from feature import ex_feature


# =========================================================
# 1) 当前候选特征名
#    必须与当前 feature.py 中 ex_feature 输出顺序严格一致
# =========================================================
FEATURE_NAMES = [

    "rho",
    "c20_re_n",
    "c20_im_n",
    "dphi_std",
    "ph",
    "pn_b3",
    "y_envelope_mean",
    "P_o",   # 时域方差
    "P_y",   # 时域偏度
    "P_k",   # 时域峰度
    "P_x",   # 前后半段包络比
    "R_HT",
    "J_HT",
    "SNRE",
    "Db",
    "Di",
    "LZC_y",
    "P_U",   # 总功率
    "P_O",   # 频域方差
    "P_Y",   # 频域偏度
    "P_K",   # 频域峰度
    "P_X",   # 前后半谱比
]


# =========================================================
# 1.1) 特征名称对照表（用于打印、保存、画图显示）
# =========================================================
# FEATURE_DISPLAY_NAMES = {
#     "SNRE": "SNRE\n信噪比估计",
#     "rho": "rho\n不圆度",
#     "c20_re_n": "c20_re_n\n归一化二阶复累积量实部",
#     "c20_im_n": "c20_im_n\n归一化二阶复累积量虚部",
#     "dphi_std": "dphi_std\n相位增量标准差",
#     "ph": "ph\n相位噪声主频带特征",
#     "pn_b3": "pn_b3\n高频段相位噪声特征",
#     "y_envelope_mean": "y_envelope_mean\n包络均值",
#     "R_HT": "R_HT\nR特征",
#     "J_HT": "J_HT\nJ特征",
#     "Db": "Db\n盒维数",
#     "Di": "Di\n信息维数",
#     "LZC_y": "LZC_y\nLZC复杂度",
#     "P_o": "P_o\n时域方差",
#     "P_y": "P_y\n时域偏度",
#     "P_k": "P_k\n时域峰度",
#     "P_x": "P_x\n前后半段包络比",
#     "P_U": "P_U\n总功率",
#     "P_O": "P_O\n频域方差",
#     "P_Y": "P_Y\n频域偏度",
#     "P_K": "P_K\n频域峰度",
#     "P_X": "P_X\n前后半谱比",
# }
FEATURE_DISPLAY_NAMES = {

    "rho": "rho\n",
    "c20_re_n": "c20_re_n\n",
    "c20_im_n": "c20_im_n\n",
    "dphi_std": "d_rms\n",
    "ph": "P_iN\n",
    "pn_b3": "P_out\n",
    "y_envelope_mean": "u_a\n",
    "P_o": "a_o\n",
    "P_y": "a_s\n",
    "P_k": "a_k\n",
    "P_x": "a_r\n",
    "R_HT": "R_HT\n",
    "J_HT": "J_HT\n",
    "SNRE": "SNRe\n",
    "Db": "Db\n",
    "Di": "Di\n",
    "LZC_y": "LZC\n",
    "P_U": "P_u\n",
    "P_O": "P_o\n",
    "P_Y": "P_s\n",
    "P_K": "P_k\n",
    "P_X": "P_r\n",
}

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


def load_or_build_feature_table(root_dir, feature_csv, force_recompute=False):
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
        df = build_feature_table(root_dir, save_csv=feature_csv)
        return df


# =========================================================
# 3) 特征名称对照表保存/输出
# =========================================================
def save_feature_name_mapping(feature_names, feature_display_names, save_csv=None, print_table=True):
    rows = []
    for idx, feat in enumerate(feature_names, 1):
        display_name = feature_display_names.get(feat, feat)
        if "\n" in display_name:
            short_name, chinese_name = display_name.split("\n", 1)
        else:
            short_name, chinese_name = feat, display_name

        rows.append({
            "index": idx,
            "feature_code": feat,
            "short_name": short_name,
            "display_name": display_name.replace("\n", " / "),
            "chinese_name": chinese_name,
        })

    df_map = pd.DataFrame(rows)

    if save_csv is not None:
        df_map.to_csv(save_csv, index=False, encoding="utf-8-sig")

    if print_table:
        print("\n========== 特征名称对照表 ==========")
        print(df_map.to_string(index=False))

    return df_map


# =========================================================
# 4) 读取 .mat 文件
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
# 5) 单个样本提特征
# =========================================================
def extract_one_feature_vector(x: np.ndarray, fs: float):
    x = np.asarray(x).squeeze()

    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape={x.shape}")

    signal_matrix = np.expand_dims(x, axis=0)

    feat = ex_feature(signal_matrix, fs)

    feat = np.asarray(feat).squeeze()

    if feat.ndim != 1:
        raise ValueError(f"feature output must be 1-D after squeeze, got shape={feat.shape}")

    if len(feat) != len(FEATURE_NAMES):
        raise ValueError(
            f"feature dimension mismatch: got {len(feat)}, expected {len(FEATURE_NAMES)}"
        )

    return feat


# =========================================================
# 6) 批量构建总特征表
# =========================================================
def build_feature_table(root_dir: str, save_csv: str = None):
    mat_files = glob.glob(os.path.join(root_dir, "**", "*.mat"), recursive=True)
    mat_files = sorted(mat_files)

    if len(mat_files) == 0:
        raise FileNotFoundError(f"No .mat files found under: {root_dir}")

    rows = []

    for i, mat_path in enumerate(mat_files, 1):
        meta = read_one_mat(mat_path)

        try:
            feat = extract_one_feature_vector(meta["x_bb1"], meta["fs"])
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
# 7) 稳定性：类内标准差 / 类内平均绝对值
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
# 8) 可分性：Fisher score
# =========================================================
def fisher_score_one_condition(df_cond: pd.DataFrame, feat: str):
    class_means = df_cond.groupby("user_id")[feat].mean()
    global_mean = df_cond[feat].mean()

    sb = ((class_means - global_mean) ** 2).sum()

    sw = 0.0
    for _, sub in df_cond.groupby("user_id"):
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
# 9) 单特征分类准确率
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
# 10) 综合汇总表
# =========================================================
def build_summary_table(cv_df, fisher_df, acc_df):
    cv_mean = cv_df.groupby("feature")["cv"].mean().rename("avg_cv")
    fisher_mean = fisher_df.groupby("feature")["fisher"].mean().rename("avg_fisher")
    acc_mean = acc_df.groupby("feature")["acc_mean"].mean().rename("avg_acc")

    summary = pd.concat([cv_mean, fisher_mean, acc_mean], axis=1).reset_index()

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
# 11) 自动优选：输出前 top_k 个特征
# =========================================================
def save_selected_features(summary_df, top_k=10, save_csv=None):
    selected_df = summary_df.head(top_k).copy()
    selected_df.insert(0, "rank", np.arange(1, len(selected_df) + 1))

    if save_csv is not None:
        selected_df.to_csv(save_csv, index=False, encoding="utf-8-sig")

    return selected_df


# =========================================================
# 12) 网格图
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

        display_name = FEATURE_DISPLAY_NAMES.get(feat, feat)

        ax.set_xlabel("Hopping rate (hop/s)")
        ax.set_ylabel(ylabel)
        ax.set_title(display_name, fontsize=10)
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

    display_labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in summary_df["feature"]]

    plt.figure(figsize=(14, 6))
    plt.bar(display_labels, summary_df["final_score"])
    plt.xlabel("Feature")
    plt.ylabel("Final Score")
    plt.title("Comprehensive Ranking of Features")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format="svg", bbox_inches="tight")

    plt.show()


# =========================================================
# 13) 主函数
# =========================================================
def main():
    root_dir = r"D:\matrixlab\exp_data_m5_5_8_snr_-5_20"
    out_dir = r"D:\matrixlab\exp_result_m5_5_8_22w_snr_-5_20"
    os.makedirs(out_dir, exist_ok=True)

    force_feature = False
    force_cv = False
    force_fisher = False
    force_acc = False
    force_summary = False

    top_k = 10

    feature_csv = os.path.join(out_dir, "feature_table.csv")
    cv_csv = os.path.join(out_dir, "cv_table.csv")
    fisher_csv = os.path.join(out_dir, "fisher_table.csv")
    acc_csv = os.path.join(out_dir, "acc_table.csv")
    summary_csv = os.path.join(out_dir, "feature_summary.csv")
    selected_csv = os.path.join(out_dir, f"selected_top{top_k}.csv")
    mapping_csv = os.path.join(out_dir, "feature_name_mapping.csv")

    # 1. 特征表
    df = load_or_build_feature_table(
        root_dir=root_dir,
        feature_csv=feature_csv,
        force_recompute=force_feature,
    )

    print("\nFeature table head:")
    print(df.head())
    print("\nShape:", df.shape)

    # 2. 输出特征名称对照表
    mapping_df = save_feature_name_mapping(
        FEATURE_NAMES,
        FEATURE_DISPLAY_NAMES,
        save_csv=mapping_csv,
        print_table=True
    )

    # 3. CV
    cv_df = load_or_compute_table(
        csv_path=cv_csv,
        compute_func=compute_cv_table,
        force_recompute=force_cv,
        df=df,
        feature_names=FEATURE_NAMES
    )

    # 4. Fisher
    fisher_df = load_or_compute_table(
        csv_path=fisher_csv,
        compute_func=compute_fisher_table,
        force_recompute=force_fisher,
        df=df,
        feature_names=FEATURE_NAMES
    )

    # 5. Accuracy
    acc_df = load_or_compute_table(
        csv_path=acc_csv,
        compute_func=compute_single_feature_acc,
        force_recompute=force_acc,
        df=df,
        feature_names=FEATURE_NAMES,
        n_splits=5
    )

    # 6. Summary
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

    # 7. 自动优选
    selected_df = save_selected_features(
        summary_df,
        top_k=top_k,
        save_csv=selected_csv
    )
    print(f"\nTop-{top_k} selected features:")
    print(selected_df)

    # 8. 画图
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
    print(f"Feature name mapping saved to: {mapping_csv}")


if __name__ == "__main__":
    main()