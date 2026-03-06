import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import scipy.io
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from extract_feature import extract_feature  # 假设你的18维特征函数保存为extract_feature.py
from feature import ex_feature
import re
def get_true_label(filename):
    m = re.search(r'_(\d+)', filename)
    if m is None:
        raise ValueError(f"无法解析标签: {filename}")
    return int(m.group(1))


# 主程序参数
folder = r'D:\matrixlab\match_tar\test29TPLB'  # 修改为你的信号路径
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
    for file in tqdm(os.listdir(folder)):
        if file.endswith('.mat'):
            path = os.path.join(folder, file)
            data = scipy.io.loadmat(path)
            if 'fil_base' in data:
                signal = data['fil_base']
                if signal.ndim > 1:
                    signal = signal.flatten()
                signal_list.append(signal)
                file_list.append(file)
                # feature_vector = extract_feature(signal[np.newaxis, :], Fs)[0]
                # features_list.append(feature_vector)
    original_signal_matrix = np.vstack(signal_list)
    feature_matrix=ex_feature(original_signal_matrix, Fs)

    X = np.array(feature_matrix)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # # 保存标准化特征和文件名列表
    # np.savez(save_path, X_scaled=X_scaled, file_list=file_list)
    print(f"特征提取完毕，保存至 {save_path}")

for i in range(len(file_list)):
    print(file_list[i])
    print(X[i])

# KMeans 五类聚类
kmeans = KMeans(n_clusters=5, random_state=3, n_init=20)
labels = kmeans.fit_predict(X_scaled)

# 构造真实标签
y_true = np.array([get_true_label(fname) - 1 for fname in file_list])
num_classes = len(np.unique(y_true))

# ===== 标签对齐（关键）=====
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

def align_kmeans_labels(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in y_pred])

labels_aligned = align_kmeans_labels(y_true, labels)

# ===== PCA 可视化 =====
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
palette = sns.color_palette("bright", num_classes)

for cls in range(num_classes):
    idx = labels_aligned == cls
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