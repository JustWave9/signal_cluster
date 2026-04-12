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

# 你的传统特征函数
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


def get_true_label(filename: str) -> int:
    device_id, _ = parse_filename(filename)
    return device_id - 1   # 转成 0~4


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
#    缓存：
#    - handcrafted feature
#    - patch_tf
#    - label / snr / run / hop_time / filename
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

            # ------- patch -------
            patch = np.squeeze(data["patch_tf"]).astype(np.float32)
            pmin, pmax = patch.min(), patch.max()
            if pmax > pmin:
                patch = (patch - pmin) / (pmax - pmin)

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
# 4. 按 run 划分训练 / 验证 / 测试
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


# =========================================================
# 5. 人工特征标准化
#    只用训练集 fit，验证/测试只 transform
# =========================================================
def normalize_handcrafted_features(split):
    scaler = StandardScaler()
    split["train"]["hand_feats"] = scaler.fit_transform(split["train"]["hand_feats"]).astype(np.float32)
    split["val"]["hand_feats"]   = scaler.transform(split["val"]["hand_feats"]).astype(np.float32)
    split["test"]["hand_feats"]  = scaler.transform(split["test"]["hand_feats"]).astype(np.float32)
    return split, scaler


# =========================================================
# 6. 按 SNR 过滤数据集
# =========================================================
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
# 7. 任务构造：
#    同一个 run、同一个时隙 support；其他时隙 query
#    -> 本质是 5-way 1-shot
# =========================================================
def sample_episode_same_run_same_slot(dataset, n_way=5, n_query=5):
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

    # 2) 这个 SNR 下有哪些 run
    candidate_runs = np.unique(runs[snrs == episode_snr])
    if len(candidate_runs) == 0:
        raise ValueError(f"SNR={episode_snr} 下没有可用 run")

    episode_run = np.random.choice(candidate_runs)

    # 3) 在这个 run 下找可用 support 时隙
    mask_run = (snrs == episode_snr) & (runs == episode_run)
    candidate_hops = np.unique(hops[mask_run])

    valid_support_hops = []
    for h in candidate_hops:
        ok = True
        for cls in selected_classes:
            idx = np.where(mask_run & (hops == h) & (y == cls))[0]
            if len(idx) < 1:
                ok = False
                break
        if ok:
            valid_support_hops.append(h)

    if len(valid_support_hops) == 0:
        raise ValueError(f"SNR={episode_snr}, run={episode_run} 下没有合法 support 时隙")

    support_hop = np.random.choice(valid_support_hops)

    support_h, support_p, support_y = [], [], []
    query_h, query_p, query_y = [], [], []

    for epi_label, cls in enumerate(selected_classes):
        # support: 同 run、同 support_hop、同类
        s_idx = np.where(
            (snrs == episode_snr) &
            (runs == episode_run) &
            (hops == support_hop) &
            (y == cls)
        )[0]

        if len(s_idx) < 1:
            raise ValueError(
                f"SNR={episode_snr}, run={episode_run}, hop={support_hop}, cls={cls} support 不足"
            )

        chosen_s = np.random.choice(s_idx, size=1, replace=False)

        # query: 同 run、同类、但 hop != support_hop
        q_idx = np.where(
            (snrs == episode_snr) &
            (runs == episode_run) &
            (hops != support_hop) &
            (y == cls)
        )[0]

        if len(q_idx) < n_query:
            raise ValueError(
                f"SNR={episode_snr}, run={episode_run}, cls={cls} query 不足，只有 {len(q_idx)} 个"
            )

        chosen_q = np.random.choice(q_idx, size=n_query, replace=False)

        support_h.append(Xh[chosen_s])
        support_p.append(Xp[chosen_s])
        support_y.extend([epi_label] * 1)

        query_h.append(Xh[chosen_q])
        query_p.append(Xp[chosen_q])
        query_y.extend([epi_label] * n_query)

    support_h = np.concatenate(support_h, axis=0)
    support_p = np.concatenate(support_p, axis=0)
    query_h   = np.concatenate(query_h, axis=0)
    query_p   = np.concatenate(query_p, axis=0)

    support_y = np.array(support_y, dtype=np.int64)
    query_y   = np.array(query_y, dtype=np.int64)

    return support_h, support_p, support_y, query_h, query_p, query_y


# =========================================================
# 8. 模型定义
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

    def encode(self, hand_x, patch_x):
        z_patch = self.patch_encoder(patch_x)
        z_hand = self.hand_encoder(hand_x)
        z = torch.cat([z_hand, z_patch], dim=1)
        z = self.fusion(z)
        z = F.normalize(z, p=2, dim=1)
        return z


# =========================================================
# 9. Proto loss（支持 CUDA + AMP）
# =========================================================
def prototypical_loss(model, support_h, support_p, support_y, query_h, query_p, query_y, device, use_amp=False):
    support_h = torch.tensor(support_h, dtype=torch.float32, device=device)
    query_h   = torch.tensor(query_h, dtype=torch.float32, device=device)

    support_p = torch.tensor(support_p, dtype=torch.float32, device=device).unsqueeze(1)
    query_p   = torch.tensor(query_p, dtype=torch.float32, device=device).unsqueeze(1)

    support_y = torch.tensor(support_y, dtype=torch.long, device=device)
    query_y   = torch.tensor(query_y, dtype=torch.long, device=device)

    with autocast(device_type="cuda", enabled=use_amp):
        z_support = model.encode(support_h, support_p)
        z_query   = model.encode(query_h, query_p)

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
# 10. 验证 / 测试
# =========================================================
@torch.no_grad()
def evaluate_episodes(model, dataset, device, n_way=5, n_query=5, num_episodes=200):
    model.eval()
    acc_list = []
    use_amp = (device.type == "cuda")

    for _ in range(num_episodes):
        support_h, support_p, support_y, query_h, query_p, query_y = \
            sample_episode_same_run_same_slot(dataset, n_way=n_way, n_query=n_query)

        _, acc = prototypical_loss(
            model,
            support_h, support_p, support_y,
            query_h, query_p, query_y,
            device,
            use_amp=use_amp
        )
        acc_list.append(acc)

    return float(np.mean(acc_list)), float(np.std(acc_list))


# =========================================================
# 11. 按 SNR 测试
# =========================================================
def evaluate_by_snr(model, dataset, device, snr_list, n_way=5, n_query=5, num_episodes=200):
    results = {}

    for snr in snr_list:
        ds_snr = filter_dataset_by_snr(dataset, snr)

        num_samples = len(ds_snr["labels"])
        if num_samples == 0:
            print(f"[跳过] SNR={snr} dB 没有样本")
            continue

        unique_classes = np.unique(ds_snr["labels"])
        if len(unique_classes) < n_way:
            print(f"[跳过] SNR={snr} dB 类别数不足，只有 {len(unique_classes)} 类")
            continue

        try:
            mean_acc, std_acc = evaluate_episodes(
                model,
                ds_snr,
                device,
                n_way=n_way,
                n_query=n_query,
                num_episodes=num_episodes
            )
            results[snr] = {
                "mean": mean_acc,
                "std": std_acc,
                "num_samples": num_samples
            }
            print(f"SNR={snr:>2} dB | few-shot acc = {mean_acc:.4f} ± {std_acc:.4f} | samples={num_samples}")
        except Exception as e:
            print(f"[失败] SNR={snr} dB -> {e}")

    return results


def plot_accuracy_vs_snr(results_dict, save_path=None):
    if len(results_dict) == 0:
        print("没有可画的 SNR 结果")
        return

    snrs = sorted(results_dict.keys())
    means = [results_dict[s]["mean"] for s in snrs]
    stds = [results_dict[s]["std"] for s in snrs]

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        snrs, means, yerr=stds,
        marker='o', linewidth=2, capsize=4
    )
    plt.xlabel("SNR (dB)")
    plt.ylabel("Few-shot Accuracy")
    plt.title("ProtoNet Few-shot Accuracy vs SNR")
    plt.grid(True)
    plt.xticks(snrs)
    plt.ylim(0, 1.05)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


# =========================================================
# 12. 训练
# =========================================================
def train_protonet(model, split, device, n_way=5, n_query=5,
                   epochs=30, episodes_per_epoch=100, val_episodes=100, lr=1e-3):
    optimizer = Adam(model.parameters(), lr=lr)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_acc = -1.0
    best_state = None
    use_amp = (device.type == "cuda")

    for epoch in range(1, epochs + 1):
        model.train()
        train_accs = []
        train_losses = []

        for _ in range(episodes_per_epoch):
            support_h, support_p, support_y, query_h, query_p, query_y = \
                sample_episode_same_run_same_slot(split["train"], n_way=n_way, n_query=n_query)

            optimizer.zero_grad(set_to_none=True)

            loss, acc = prototypical_loss(
                model,
                support_h, support_p, support_y,
                query_h, query_p, query_y,
                device,
                use_amp=use_amp
            )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            train_accs.append(acc)

        val_mean, val_std = evaluate_episodes(
            model, split["val"], device,
            n_way=n_way, n_query=n_query,
            num_episodes=val_episodes
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={np.mean(train_losses):.4f} | "
            f"train_acc={np.mean(train_accs):.4f} | "
            f"val_acc={val_mean:.4f} ± {val_std:.4f}"
        )

        if val_mean > best_val_acc:
            best_val_acc = val_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# =========================================================
# 13. 主程序
# =========================================================
def main():
    # -----------------------------
    # 路径
    # -----------------------------
    base_dir = r"D:\matrixlab\match_tar\p5"
    cache_path = os.path.join(base_dir, "fusion_proto_cache_same_run_same_slot.npz")

    # 当前数据：0~9
    allowed_snrs = set(range(0, 11))
    test_snr_list = sorted(list(allowed_snrs))

    # -----------------------------
    # few-shot 设置
    # -----------------------------
    n_way = 5
    n_shot = 1
    n_query = 5

    # -----------------------------
    # 读取数据
    # -----------------------------
    data_dict = build_or_load_cache(
        base_dir,
        cache_path,
        Fs=1e7,
        allowed_snrs=allowed_snrs
    )
    split = split_by_run(data_dict)
    split, scaler = normalize_handcrafted_features(split)

    print("Train size:", len(split["train"]["labels"]))
    print("Val size  :", len(split["val"]["labels"]))
    print("Test size :", len(split["test"]["labels"]))

    hand_dim = split["train"]["hand_feats"].shape[1]
    patch_shape = split["train"]["patches"].shape[1:]
    print("Handcrafted feature dim:", hand_dim)
    print("Patch shape:", patch_shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version in torch:", torch.version.cuda)

    model = FusionProtoNet(
        hand_dim=hand_dim,
        patch_dim=64,
        hand_emb_dim=64,
        fusion_dim=64
    ).to(device)

    # -----------------------------
    # 训练
    # -----------------------------
    model = train_protonet(
        model,
        split,
        device,
        n_way=n_way,
        n_query=n_query,
        epochs=30,
        episodes_per_epoch=100,
        val_episodes=100,
        lr=1e-3
    )

    # -----------------------------
    # 整体测试
    # -----------------------------
    test_mean, test_std = evaluate_episodes(
        model, split["test"], device,
        n_way=n_way, n_query=n_query,
        num_episodes=300
    )

    print("\n==============================")
    print(f"{n_way}-way {n_shot}-shot ProtoNet")
    print("Support: same run + same slot")
    print("Query  : same run + other slots")
    print(f"Overall test episode accuracy = {test_mean:.4f} ± {test_std:.4f}")
    print("==============================")

    # -----------------------------
    # 按 SNR 分组测试
    # -----------------------------
    snr_results = evaluate_by_snr(
        model,
        split["test"],
        device,
        snr_list=test_snr_list,
        n_way=n_way,
        n_query=n_query,
        num_episodes=200
    )

    plot_accuracy_vs_snr(
        snr_results,
        save_path=os.path.join(base_dir, "fewshot_accuracy_vs_snr.png")
    )

    rows = []
    for snr in sorted(snr_results.keys()):
        rows.append({
            "snr": snr,
            "fewshot_acc_mean": snr_results[snr]["mean"],
            "fewshot_acc_std": snr_results[snr]["std"],
            "num_samples": snr_results[snr]["num_samples"]
        })

    if len(rows) > 0:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(base_dir, "fewshot_accuracy_vs_snr.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"SNR测试结果已保存：{csv_path}")

    # -----------------------------
    # 保存模型
    # -----------------------------
    model_path = os.path.join(base_dir, "fusion_protonet_same_run_same_slot.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存：{model_path}")


if __name__ == "__main__":
    main()