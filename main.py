import os
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

def get_true_label(filename):
    return int(filename.split('_')[1])

# 主程序参数
folder = r'D:\matrixlab\match_tar\test22TPLB'  # 修改为你的信号路径
Fs = 1e7  # 采样率
save_path = "test3w.npz"  # 保存标准化特征文件名

if os.path.exists(save_path):
    print("检测到已保存特征，直接加载...")
    data = np.load(save_path, allow_pickle=True)
    X_scaled = data["X_scaled"]
    file_list = data["file_list"].tolist()
else:
    print("未检测到保存文件，开始提取特征...")
    features_list = []
    file_list = []


    for file in tqdm(os.listdir(folder)):
        if file.endswith('.mat'):
            path = os.path.join(folder, file)
            data = scipy.io.loadmat(path)
            if 'baseband_signal' in data:
                signal = data['baseband_signal']
                if signal.ndim > 1:
                    signal = signal.flatten()
                feature_vector = extract_feature(signal[np.newaxis, :], Fs)[0]
                features_list.append(feature_vector)
                file_list.append(file)

    X = np.array(features_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

# # 保存标准化特征和文件名列表
np.savez(save_path, X_scaled=X_scaled, file_list=file_list)
print(f"特征提取完毕，保存至 {save_path}")

for i in range(len(file_list)):
    print(file_list[i])
    print(X_scaled[i])


# KMeans聚类
kmeans = KMeans(n_clusters=2, random_state=3)
labels = kmeans.fit_predict(X_scaled)

# 输出聚类结果
print("文件名 → 聚类类别")
for fname, label in zip(file_list, labels):
    print(f"{fname} → 类别 {label}")

# 构造真实标签
y_true = [get_true_label(fname)-1 for fname in file_list]
acc1 = accuracy_score(y_true, labels)
acc2 = accuracy_score(y_true, 1 - labels)
if acc2 > acc1:
    labels = 1 - labels  # 标签对齐

# 根据真实标签绘制散点图
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
palette = sns.color_palette("bright", 2)

for label in np.unique(y_true):
    idx = np.array(y_true) == label
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1],
                label=f'True Class {label}', alpha=0.7, s=50, c=[palette[label]])

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title(" PCA ")
plt.legend()
plt.grid(True)
plt.show()


# 混淆矩阵
cm = confusion_matrix(y_true, labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("KMeans Confusion Matrix")
plt.tight_layout()
plt.show()

# 分类报告
print(classification_report(y_true, labels, target_names=["Class 0", "Class 1"]))



# import os
# import numpy as np
# import scipy.io
# from scipy.fft import fft
# from scipy.signal import correlate, find_peaks
# import scipy.stats as stats
# import pywt
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# def extract_features(signal, Fs):
#     signal = np.asarray(signal).flatten()
#     abs_signal = np.abs(signal)  # 复信号转为模值
#     L = len(signal)
#     T = 1 / Fs
#     features = {}
#
#     # 一、统计特征（基于模值）
#     features['max'] = np.max(abs_signal)
#     features['min'] = np.min(abs_signal)
#     features['mean'] = np.mean(abs_signal)
#     features['median'] = np.median(abs_signal)
#     features['skewness'] = stats.skew(abs_signal, bias=False)
#     features['kurtosis'] = stats.kurtosis(abs_signal, fisher=False)
#     features['iqr'] = np.percentile(abs_signal, 75) - np.percentile(abs_signal, 25)
#     features['mad_mean'] = np.mean(np.abs(abs_signal - np.mean(abs_signal)))
#     features['mad_median'] = np.median(np.abs(abs_signal - np.median(abs_signal)))
#     features['rms'] = np.sqrt(np.mean(abs_signal**2))
#     features['std'] = np.std(abs_signal)
#     features['var'] = np.var(abs_signal)
#     features['percentile_50'] = np.sum(abs_signal <= np.percentile(abs_signal, 50)) / len(abs_signal)
#
#     # 二、频谱特征（模值）
#     fft_val = fft(signal)
#     P2 = np.abs(fft_val / L)
#     P1 = P2[:L // 2 + 1]
#     P1[1:-1] *= 2
#     features['fft_mean'] = np.mean(P1)
#     features['fft_max'] = np.max(P1)
#     features['fft_median'] = np.median(P1)
#     peaks, _ = find_peaks(P1)
#     features['fft_base'] = peaks[0] * Fs / L if len(peaks) > 0 else 0
#
#     # 三、小波特征（模值）
#     coeffs = pywt.wavedec(abs_signal, 'db1', level=5)
#     features['wavelet_abs_mean'] = np.mean([np.mean(np.abs(c)) for c in coeffs])
#     features['wavelet_std'] = np.mean([np.std(c) for c in coeffs])
#     # features['wavelet_var'] = np.mean([np.var(c) for c in coeffs])
#
#     # 四、差分特征（模值）
#     diff = np.diff(abs_signal)
#     features['diff_mean'] = np.mean(diff)
#     # features['diff_abs_mean'] = np.mean(np.abs(diff))
#     features['diff_median'] = np.median(diff)
#     features['diff_abs_median'] = np.median(np.abs(diff))
#     # features['diff_sum'] = np.sum(np.abs(diff))
#
#     # 五、熵（模值）
#     hist, _ = np.histogram(abs_signal, bins=256, density=True)
#     hist = hist[hist > 0]
#     features['entropy'] = -np.sum(hist * np.log2(hist))
#
#     # 六、几何特征（模值）
#     features['x_dist_peak_valley'] = np.abs(np.argmax(abs_signal) - np.argmin(abs_signal)) * T
#     features['area'] = np.sum(abs_signal)
#     # features['num_max_peaks'] = len(find_peaks(abs_signal)[0])
#     # features['num_min_peaks'] = len(find_peaks(-abs_signal)[0])
#     features['zero_cross_rate'] = np.sum((abs_signal[:-1] * abs_signal[1:] < 0) | (abs_signal[:-1] == 0))
#
#     return features
#
# # 从文件名中提取真实标签（第一个下划线后的数字）
# def get_true_label(filename):
#     return int(filename.split('_')[1])
#
#
# # === 主程序 ===
# folder = (r'D:\matrixlab\match_tar\test4m'
#           r'')  # 修改为你的信号路径
# Fs = 1e7  # 采样率
#
# features_list = []
# file_list = []
#
# for file in os.listdir(folder):
#     if file.endswith('.mat'):
#         path = os.path.join(folder, file)
#         data = scipy.io.loadmat(path)
#         if 'baseband_signal' in data:
#             signal = data['baseband_signal'].flatten()
#             features = extract_features(signal, Fs)
#             features_list.append(list(features.values()))
#             file_list.append(file)
#             # ✅ 打印每个特征
#             print(f"\n📂 文件: {file}")
#             for k, v in features.items():
#                 print(f"  {k:<25}: {v:.4e}")
#
# # 转成数组并标准化
# X = np.array(features_list)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # KMeans聚类
# kmeans = KMeans(n_clusters=2, random_state=0)
# labels = kmeans.fit_predict(X_scaled)
#
# # 输出聚类结果
# print("文件名 → 聚类类别")
# for fname, label in zip(file_list, labels):
#     print(f"{fname} → 类别 {label}")
#
# # 构造真实标签列表
# y_true = [get_true_label(fname)-1 for fname in file_list]
# # 对齐KMeans输出标签（防止0/1顺序颠倒）
# acc1 = accuracy_score(y_true, labels)
# acc2 = accuracy_score(y_true, 1 - labels)
# if acc2 > acc1:
#     labels = 1 - labels  # 标签反转对齐
#
# # 混淆矩阵
# cm = confusion_matrix(y_true, labels)
#
# # 可视化混淆矩阵
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#             xticklabels=["Pred 0", "Pred 1"],
#             yticklabels=["True 0", "True 1"])
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("KMeans Confusion Matrix")
# plt.tight_layout()
# plt.show()
#
# # 分类报告
# print(classification_report(y_true, labels, target_names=["Class 0", "Class 1"]))
