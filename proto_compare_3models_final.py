import os
import re
import random
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.amp import autocast, GradScaler

from feature import ex_feature


# =========================================================
# 1. 基础设置
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

torch.backends.cudnn.benchmark = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# =========================================================
# 2. 文件名 / 文件夹名解析
# =========================================================
def parse_filename(filename: str):
    """
    文件名示例:
    signal17_1_17_176001.000000_186000.000000_3500000.000000.mat

    parts[1] -> device_id
    parts[2] -> hop_time
    """
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    if len(parts) < 3:
        raise ValueError(f"文件名格式不符合预期: {filename}")

    device_id = int(parts[1])
    hop_time = int(parts[2])
    return device_id, hop_time


def parse_folder_meta(folder_name: str):
    """
    文件夹示例:
    test37rNBP_snr3_run2_bb1
    """
    m = re.search(r"snr(-?\d+)_run(\d+)_bb1", folder_name)
    if m is None:
        raise ValueError(f"无法解析文件夹名: {folder_name}")
    snr = int(m.group(1))
    run = int(m.group(2))
    return snr, run


# =========================================================
# 3. 构建 / 读取缓存
# =========================================================
def build_or_load_cache(base_dir, cache_path, Fs=1e7, allowed_snrs=None):
    if os.path.exists(cache_path):
        print(f"[缓存] 直接加载: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return {
            "hand_feats": data["hand_feats"],
            "patches": data["patches"],
            "labels": data["labels"],
            "snrs": data["snrs"],
            "runs": data["runs"],
            "hop_times": data["hop_times"],
            "filenames": data["filenames"].tolist(),
        }

    print("[缓存] 未找到缓存，开始扫描并提取特征...")

    folder_list = []
    for d in os.listdir(base_dir):
        full_path = os.path.join(base_dir, d)
        if not (os.path.isdir(full_path) and "test37rNBP_snr" in d and "_bb1" in d):
            continue

        try:
            snr, run = parse_folder_meta(d)
        except Exception:
            continue

        if allowed_snrs is not None and snr not in allowed_snrs:
            continue

        folder_list.append(d)

    folder_list = sorted(folder_list)
    if len(folder_list) == 0:
        raise RuntimeError(f"{base_dir} 下没有找到匹配的数据文件夹")

    hand_feats = []
    patches = []
    labels = []
    snrs = []
    runs = []
    hop_times = []
    filenames = []

    for folder_name in folder_list:
        folder = os.path.join(base_dir, folder_name)
        snr, run = parse_folder_meta(folder_name)

        mat_files = sorted([f for f in os.listdir(folder) if f.endswith(".mat")])

        for fname in tqdm(mat_files, desc=f"处理 {folder_name}"):
            path = os.path.join(folder, fname)
            data = scipy.io.loadmat(path)

            if ("x_bb1" not in data) or ("patch_tf" not in data):
                continue

            # ------- 传统特征 -------
            signal = np.squeeze(data["x_bb1"])
            feat = ex_feature(signal[np.newaxis, :], Fs)[0]
            feat = np.asarray(feat, dtype=np.float32)
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

            # ------- patch -------
            patch = np.squeeze(data["patch_tf"]).astype(np.float32)
            pmin, pmax = patch.min(), patch.max()
            if pmax > pmin:
                patch = (patch - pmin) / (pmax - pmin)
            patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)

            device_id, hop_time = parse_filename(fname)
            label = device_id - 1

            hand_feats.append(feat)
            patches.append(patch)
            labels.append(label)
            snrs.append(snr)
            runs.append(run)
            hop_times.append(hop_time)
            filenames.append(fname)

    hand_feats = np.array(hand_feats, dtype=np.float32)
    patches = np.array(patches, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    snrs = np.array(snrs, dtype=np.int64)
    runs = np.array(runs, dtype=np.int64)
    hop_times = np.array(hop_times, dtype=np.int64)

    np.savez(
        cache_path,
        hand_feats=hand_feats,
        patches=patches,
        labels=labels,
        snrs=snrs,
        runs=runs,
        hop_times=hop_times,
        filenames=np.array(filenames, dtype=object)
    )

    print(f"[缓存] 已保存到: {cache_path}")

    return {
        "hand_feats": hand_feats,
        "patches": patches,
        "labels": labels,
        "snrs": snrs,
        "runs": runs,
        "hop_times": hop_times,
        "filenames": filenames,
    }


# =========================================================
# 4. 数据划分与预处理
# =========================================================
def split_by_run(data_dict):
    hand_feats = data_dict["hand_feats"]
    patches = data_dict["patches"]
    labels = data_dict["labels"]
    runs = data_dict["runs"]
    snrs = data_dict["snrs"]
    hop_times = data_dict["hop_times"]

    train_mask = np.isin(runs, [1, 2, 3])
    val_mask   = np.isin(runs, [4])
    test_mask  = np.isin(runs, [5])

    split = {}
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        split[name] = {
            "hand_feats": hand_feats[mask],
            "patches": patches[mask],
            "labels": labels[mask],
            "runs": runs[mask],
            "snrs": snrs[mask],
            "hop_times": hop_times[mask],
        }
    return split


def normalize_handcrafted_features(split):
    scaler = StandardScaler()
    split["train"]["hand_feats"] = scaler.fit_transform(split["train"]["hand_feats"]).astype(np.float32)
    split["val"]["hand_feats"]   = scaler.transform(split["val"]["hand_feats"]).astype(np.float32)
    split["test"]["hand_feats"]  = scaler.transform(split["test"]["hand_feats"]).astype(np.float32)
    return split, scaler


def filter_dataset_by_snr(dataset, target_snr):
    mask = (dataset["snrs"] == target_snr)
    return {
        "hand_feats": dataset["hand_feats"][mask],
        "patches": dataset["patches"][mask],
        "labels": dataset["labels"][mask],
        "runs": dataset["runs"][mask],
        "snrs": dataset["snrs"][mask],
        "hop_times": dataset["hop_times"][mask],
    }


# =========================================================
# 5. 在线任务构造：
#    support slot vs query slot
# =========================================================
def sample_episode_support_query_slots(dataset, n_way=5):
    """
    在线时隙级小样本任务：
    - support set: 同一个 run 的某一个时隙
    - query set:   同一个 run 的另一个时隙
    - 每个时隙里每类 1 个样本

    本质上是 5-way 1-shot 1-query
    """
    Xh = dataset["hand_feats"]
    Xp = dataset["patches"]
    y  = dataset["labels"]
    snrs = dataset["snrs"]
    runs = dataset["runs"]
    hops = dataset["hop_times"]

    classes = np.unique(y)
    if len(classes) < n_way:
        raise ValueError(f"类别数不足: 需要 {n_way} 类，但只有 {len(classes)} 类")

    selected_classes = np.random.choice(classes, size=n_way, replace=False)

    # 1) 随机选一个 SNR
    episode_snr = np.random.choice(np.unique(snrs))

    # 2) 随机选一个 run
    candidate_runs = np.unique(runs[snrs == episode_snr])
    if len(candidate_runs) == 0:
        raise ValueError(f"SNR={episode_snr} 下没有可用 run")
    episode_run = np.random.choice(candidate_runs)

    mask_run = (snrs == episode_snr) & (runs == episode_run)
    candidate_hops = np.unique(hops[mask_run])

    # 3) 找所有合法时隙：该时隙下 selected_classes 都齐全
    valid_hops = []
    for h in candidate_hops:
        ok = True
        for cls in selected_classes:
            idx = np.where(mask_run & (hops == h) & (y == cls))[0]
            if len(idx) < 1:
                ok = False
                break
        if ok:
            valid_hops.append(h)

    if len(valid_hops) < 2:
        raise ValueError(f"SNR={episode_snr}, run={episode_run} 下合法时隙不足 2 个")

    # 4) support/query 选两个不同的时隙
    support_hop, query_hop = np.random.choice(valid_hops, size=2, replace=False)

    support_h, support_p, support_y = [], [], []
    query_h, query_p, query_y = [], [], []

    for epi_label, cls in enumerate(selected_classes):
        # support
        s_idx = np.where(
            (snrs == episode_snr) &
            (runs == episode_run) &
            (hops == support_hop) &
            (y == cls)
        )[0]
        chosen_s = np.random.choice(s_idx, size=1, replace=False)

        # query
        q_idx = np.where(
            (snrs == episode_snr) &
            (runs == episode_run) &
            (hops == query_hop) &
            (y == cls)
        )[0]
        chosen_q = np.random.choice(q_idx, size=1, replace=False)

        support_h.append(Xh[chosen_s])
        support_p.append(Xp[chosen_s])
        support_y.extend([epi_label])

        query_h.append(Xh[chosen_q])
        query_p.append(Xp[chosen_q])
        query_y.extend([epi_label])

    support_h = np.concatenate(support_h, axis=0)
    support_p = np.concatenate(support_p, axis=0)
    query_h   = np.concatenate(query_h, axis=0)
    query_p   = np.concatenate(query_p, axis=0)

    support_y = np.array(support_y, dtype=np.int64)
    query_y   = np.array(query_y, dtype=np.int64)

    meta_info = {
        "episode_snr": int(episode_snr),
        "episode_run": int(episode_run),
        "support_hop": int(support_hop),
        "query_hop": int(query_hop),
    }

    return support_h, support_p, support_y, query_h, query_p, query_y, meta_info


# =========================================================
# 6. 三个模型
# =========================================================
class PatchEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class HandcraftedEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class HandOnlyProtoNet(nn.Module):
    def __init__(self, hand_dim, emb_dim=64):
        super().__init__()
        self.hand_encoder = HandcraftedEncoder(in_dim=hand_dim, out_dim=emb_dim)

    def encode(self, hand_x=None, patch_x=None):
        z = self.hand_encoder(hand_x)
        z = F.normalize(z, p=2, dim=1)
        return z


class PatchOnlyProtoNet(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.patch_encoder = PatchEncoder(out_dim=emb_dim)

    def encode(self, hand_x=None, patch_x=None):
        z = self.patch_encoder(patch_x)
        z = F.normalize(z, p=2, dim=1)
        return z


class FusionProtoNet(nn.Module):
    def __init__(self, hand_dim, patch_dim=64, hand_emb_dim=64, fusion_dim=64):
        super().__init__()
        self.patch_encoder = PatchEncoder(out_dim=patch_dim)
        self.hand_encoder = HandcraftedEncoder(in_dim=hand_dim, out_dim=hand_emb_dim)

        self.fusion = nn.Sequential(
            nn.Linear(patch_dim + hand_emb_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, fusion_dim)
        )

    def encode(self, hand_x=None, patch_x=None):
        z_patch = self.patch_encoder(patch_x)
        z_hand = self.hand_encoder(hand_x)
        z = torch.cat([z_hand, z_patch], dim=1)
        z = self.fusion(z)
        z = F.normalize(z, p=2, dim=1)
        return z


# =========================================================
# 7. Proto loss
# =========================================================
def prototypical_loss(model, support_h, support_p, support_y, query_h, query_p, query_y,
                      device, use_amp=False, model_type="fusion"):
    support_h = torch.tensor(support_h, dtype=torch.float32, device=device)
    query_h   = torch.tensor(query_h, dtype=torch.float32, device=device)

    support_p = torch.tensor(support_p, dtype=torch.float32, device=device).unsqueeze(1)
    query_p   = torch.tensor(query_p, dtype=torch.float32, device=device).unsqueeze(1)

    support_y = torch.tensor(support_y, dtype=torch.long, device=device)
    query_y   = torch.tensor(query_y, dtype=torch.long, device=device)

    with autocast(device_type="cuda", enabled=use_amp):
        if model_type == "hand":
            z_support = model.encode(hand_x=support_h, patch_x=None)
            z_query   = model.encode(hand_x=query_h, patch_x=None)
        elif model_type == "patch":
            z_support = model.encode(hand_x=None, patch_x=support_p)
            z_query   = model.encode(hand_x=None, patch_x=query_p)
        else:
            z_support = model.encode(hand_x=support_h, patch_x=support_p)
            z_query   = model.encode(hand_x=query_h, patch_x=query_p)

        n_way = len(torch.unique(support_y))

        prototypes = []
        for c in range(n_way):
            prototypes.append(z_support[support_y == c].mean(dim=0))
        prototypes = torch.stack(prototypes, dim=0)

        dists = torch.cdist(z_query, prototypes, p=2) ** 2
        logits = -dists
        loss = F.cross_entropy(logits, query_y)

    pred = logits.argmax(dim=1)
    acc = (pred == query_y).float().mean().item()

    return loss, acc


# =========================================================
# 8. 训练 / 验证 / 测试
# =========================================================
@torch.no_grad()
def evaluate_episodes(model, dataset, device, n_way=5, num_episodes=200, model_type="fusion"):
    model.eval()
    acc_list = []
    use_amp = (device.type == "cuda")

    for _ in range(num_episodes):
        support_h, support_p, support_y, query_h, query_p, query_y, _ = \
            sample_episode_support_query_slots(dataset, n_way=n_way)

        _, acc = prototypical_loss(
            model,
            support_h, support_p, support_y,
            query_h, query_p, query_y,
            device,
            use_amp=use_amp,
            model_type=model_type
        )
        acc_list.append(acc)

    acc_arr = np.array(acc_list, dtype=np.float32)
    return {
        "mean": float(acc_arr.mean()),
        "max": float(acc_arr.max()),
        "min": float(acc_arr.min()),
    }


def train_protonet(model, split, device, n_way=5,
                   epochs=30, episodes_per_epoch=200, val_episodes=100, lr=1e-3,
                   model_type="fusion"):
    optimizer = Adam(model.parameters(), lr=lr)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_acc = -1.0
    best_state = None
    use_amp = (device.type == "cuda")

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_acc_mean": [],
        "val_acc_max": [],
        "val_acc_min": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_accs = []
        train_losses = []

        for _ in range(episodes_per_epoch):
            support_h, support_p, support_y, query_h, query_p, query_y, _ = \
                sample_episode_support_query_slots(split["train"], n_way=n_way)

            optimizer.zero_grad(set_to_none=True)

            loss, acc = prototypical_loss(
                model,
                support_h, support_p, support_y,
                query_h, query_p, query_y,
                device,
                use_amp=use_amp,
                model_type=model_type
            )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            train_accs.append(acc)

        train_loss_mean = float(np.mean(train_losses))
        train_acc_mean = float(np.mean(train_accs))

        val_result = evaluate_episodes(
            model, split["val"], device,
            n_way=n_way,
            num_episodes=val_episodes,
            model_type=model_type
        )

        val_mean = val_result["mean"]
        val_max  = val_result["max"]
        val_min  = val_result["min"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss_mean)
        history["train_acc"].append(train_acc_mean)
        history["val_acc_mean"].append(val_mean)
        history["val_acc_max"].append(val_max)
        history["val_acc_min"].append(val_min)

        print(
            f"[{model_type}] Epoch {epoch:02d} | "
            f"train_loss={train_loss_mean:.4f} | "
            f"train_acc={train_acc_mean:.4f} | "
            f"val_acc_mean={val_mean:.4f} | "
            f"val_acc_range=[{val_min:.4f}, {val_max:.4f}]"
        )

        if val_mean > best_val_acc:
            best_val_acc = val_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_by_snr(model, dataset, device, snr_list, n_way=5,
                    num_episodes=200, model_type="fusion"):
    results = {}

    for snr in snr_list:
        ds_snr = filter_dataset_by_snr(dataset, snr)
        num_samples = len(ds_snr["labels"])

        if num_samples == 0:
            print(f"[{model_type}] [跳过] SNR={snr} dB 没有样本")
            continue

        unique_classes = np.unique(ds_snr["labels"])
        if len(unique_classes) < n_way:
            print(f"[{model_type}] [跳过] SNR={snr} dB 类别数不足")
            continue

        try:
            acc_result = evaluate_episodes(
                model,
                ds_snr,
                device,
                n_way=n_way,
                num_episodes=num_episodes,
                model_type=model_type
            )
            results[snr] = {
                "mean": acc_result["mean"],
                "max": acc_result["max"],
                "min": acc_result["min"],
                "num_samples": num_samples
            }
            print(
                f"[{model_type}] SNR={snr:>2} dB | "
                f"acc_mean = {acc_result['mean']:.4f} | "
                f"range = [{acc_result['min']:.4f}, {acc_result['max']:.4f}]"
            )
        except Exception as e:
            print(f"[{model_type}] [失败] SNR={snr} dB -> {e}")

    return results


# =========================================================
# 9. 画图函数
# =========================================================
def plot_accuracy_vs_snr_compare(result_dict_all, save_path=None):
    plt.figure(figsize=(9, 5.5))

    style_map = {
        "hand": ("s", "Handcrafted only"),
        "patch": ("^", "Patch only"),
        "fusion": ("o", "Fusion")
    }

    for key in ["hand", "patch", "fusion"]:
        if key not in result_dict_all or len(result_dict_all[key]) == 0:
            continue

        snrs = sorted(result_dict_all[key].keys())
        means = [result_dict_all[key][s]["mean"] for s in snrs]
        marker, label = style_map[key]

        plt.plot(snrs, means, marker=marker, linewidth=2, label=label)

    all_snrs = sorted(list(set().union(*[set(v.keys()) for v in result_dict_all.values() if len(v) > 0])))
    if len(all_snrs) > 0:
        plt.xticks(all_snrs)

    plt.xlabel("SNR (dB)")
    plt.ylabel("Few-shot Accuracy")
    plt.title("Few-shot Accuracy vs SNR")
    plt.grid(True)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", format="svg")

    plt.show()


def plot_train_history(history, model_type, save_path=None):
    epochs = history["epoch"]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    line1, = ax1.plot(epochs, history["train_loss"], marker='o', linewidth=2, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.grid(True)

    ax2 = ax1.twinx()
    line2, = ax2.plot(epochs, history["train_acc"], marker='s', linewidth=2, label="Train Acc")
    line3, = ax2.plot(epochs, history["val_acc_mean"], marker='^', linewidth=2, label="Val Acc")

    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.0)

    handles = [line1, line2, line3]
    labels = ["Train Loss", "Train Acc", "Val Acc"]
    ax1.legend(handles, labels, loc="best")

    plt.title(f"Training History - {model_type}")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", format="svg")

    plt.show()


def plot_val_acc_compare(histories, save_path=None):
    plt.figure(figsize=(8.5, 5.5))

    style_map = {
        "hand": ("s", "Handcrafted only"),
        "patch": ("^", "Patch only"),
        "fusion": ("o", "Fusion")
    }

    for key in ["hand", "patch", "fusion"]:
        if key not in histories:
            continue

        hist = histories[key]
        marker, label = style_map[key]

        epochs = hist["epoch"]
        means  = hist["val_acc_mean"]

        plt.plot(epochs, means, marker=marker, linewidth=2, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Comparison")
    plt.grid(True)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", format="svg")

    plt.show()


# =========================================================
# 10. 模型工厂
# =========================================================
def build_model(model_type, hand_dim, device):
    if model_type == "hand":
        model = HandOnlyProtoNet(hand_dim=hand_dim, emb_dim=64)
    elif model_type == "patch":
        model = PatchOnlyProtoNet(emb_dim=64)
    elif model_type == "fusion":
        model = FusionProtoNet(hand_dim=hand_dim, patch_dim=64, hand_emb_dim=64, fusion_dim=64)
    else:
        raise ValueError(f"未知 model_type: {model_type}")

    return model.to(device)


# =========================================================
# 11. 主程序
# =========================================================
def main():
    base_dir = r"D:\matrixlab\match_tar\p5"
    cache_path = os.path.join(base_dir, "proto_compare_3models_online_slots_cache.npz")

    # 自动扫描已有 SNR；如需筛选，可改成 allowed_snrs = {0,1,2,...}
    allowed_snrs = None

    data_dict = build_or_load_cache(
        base_dir,
        cache_path,
        Fs=1e7,
        allowed_snrs=allowed_snrs
    )
    split = split_by_run(data_dict)
    split, scaler = normalize_handcrafted_features(split)

    available_test_snrs = sorted(np.unique(split["test"]["snrs"]).tolist())

    print("Train size:", len(split["train"]["labels"]))
    print("Val size  :", len(split["val"]["labels"]))
    print("Test size :", len(split["test"]["labels"]))
    print("Available test SNRs:", available_test_snrs)

    hand_dim = split["train"]["hand_feats"].shape[1]
    patch_shape = split["train"]["patches"].shape[1:]
    print("Handcrafted feature dim:", hand_dim)
    print("Patch shape:", patch_shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version in torch:", torch.version.cuda)

    n_way = 5
    n_shot = 1
    n_query = 1

    all_results = {}
    all_histories = {}
    summary_rows = []

    for model_type in ["hand", "patch", "fusion"]:
    # for model_type in ["fusion"]:
        print("\n" + "=" * 60)
        print(f"Start training model: {model_type}")
        print("=" * 60)

        model = build_model(model_type, hand_dim, device)

        model, history = train_protonet(
            model,
            split,
            device,
            n_way=n_way,
            epochs=30,
            episodes_per_epoch=100,
            val_episodes=100,
            lr=1e-3,
            model_type=model_type
        )

        all_histories[model_type] = history

        # 单模型训练曲线
        plot_train_history(
            history,
            model_type=model_type,
            save_path=os.path.join(base_dir, f"{model_type}_train_history.svg")
        )

        # 保存训练历史 CSV
        hist_df = pd.DataFrame(history)
        hist_csv = os.path.join(base_dir, f"{model_type}_train_history.csv")
        hist_df.to_csv(hist_csv, index=False, encoding="utf-8-sig")
        print(f"训练历史已保存：{hist_csv}")

        # 整体测试
        test_result = evaluate_episodes(
            model, split["test"], device,
            n_way=n_way,
            num_episodes=300,
            model_type=model_type
        )

        print("\n------------------------------")
        print(f"Model: {model_type}")
        print(f"{n_way}-way {n_shot}-shot {n_query}-query ProtoNet")
        print("Support: same run + one support slot")
        print("Query  : same run + one other slot")
        print(f"Overall test acc_mean = {test_result['mean']:.4f}")
        print(f"Overall test range    = [{test_result['min']:.4f}, {test_result['max']:.4f}]")
        print("------------------------------")

        summary_rows.append({
            "model": model_type,
            "overall_test_acc_mean": test_result["mean"],
            "overall_test_acc_max": test_result["max"],
            "overall_test_acc_min": test_result["min"]
        })

        # 按 SNR 测试
        snr_results = evaluate_by_snr(
            model,
            split["test"],
            device,
            snr_list=available_test_snrs,
            n_way=n_way,
            num_episodes=200,
            model_type=model_type
        )
        all_results[model_type] = snr_results

        # 保存模型
        model_path = os.path.join(base_dir, f"{model_type}_protonet_online_slots.pth")
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存：{model_path}")

    # 三模型 SNR 对比图
    plot_accuracy_vs_snr_compare(
        all_results,
        save_path=os.path.join(base_dir, "fewshot_accuracy_vs_snr_compare_3models.svg")
    )

    # 三模型 val_acc 对比图
    plot_val_acc_compare(
        all_histories,
        save_path=os.path.join(base_dir, "val_acc_compare_3models.svg")
    )

    # 保存整体测试总结
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(base_dir, "fewshot_overall_compare_3models.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"整体测试对比已保存：{summary_csv}")

    # 保存按 SNR 结果
    snr_rows = []
    for model_type, result_dict in all_results.items():
        for snr in sorted(result_dict.keys()):
            snr_rows.append({
                "model": model_type,
                "snr": snr,
                "fewshot_acc_mean": result_dict[snr]["mean"],
                "fewshot_acc_max": result_dict[snr]["max"],
                "fewshot_acc_min": result_dict[snr]["min"],
                "num_samples": result_dict[snr]["num_samples"]
            })

    if len(snr_rows) > 0:
        snr_df = pd.DataFrame(snr_rows)
        snr_csv = os.path.join(base_dir, "fewshot_accuracy_vs_snr_compare_3models.csv")
        snr_df.to_csv(snr_csv, index=False, encoding="utf-8-sig")
        print(f"按 SNR 对比已保存：{snr_csv}")


if __name__ == "__main__":
    main()