import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import hilbert, firwin, lfilter
from scipy.signal.windows import boxcar
from tqdm import tqdm
import logging


def estimate_main_bandwidth(signal, fs, energy_ratio=0.95):
    N = len(signal)
    freqs = fftfreq(N, d=1 / fs)
    spectrum = np.abs(fft(signal)) ** 2
    spectrum = fftshift(spectrum)
    freqs = fftshift(freqs)
    total_energy = np.sum(spectrum)
    sorted_indices = np.argsort(spectrum)[::-1]
    cumulative_energy = 0
    band_indices = []
    for idx in sorted_indices:
        cumulative_energy += spectrum[idx]
        band_indices.append(idx)
        if cumulative_energy >= energy_ratio * total_energy:
            break
    freq_band = freqs[band_indices]
    bandwidth = np.max(freq_band) - np.min(freq_band)
    return abs(bandwidth)


def extract_feature(original_signal_matrix, fs, task_index_Fea_Ext_Cal=None, queue_Fea_Ext_Cal_progress=None):
    number_of_data, signal_len = original_signal_matrix.shape
    raw_data = original_signal_matrix.T  # shape: [len, N]

    logging.info(f"\n开始特征提取，输入时域矩阵形状：{original_signal_matrix.shape}，采样率：{fs}Hz")

    # 初始化特征数组
    SNRE = np.zeros(number_of_data, dtype=complex)
    ph = np.zeros(number_of_data, dtype=complex)
    y_envelope_mean = np.zeros(number_of_data)
    R_HT = np.zeros(number_of_data)
    J_HT = np.zeros(number_of_data)
    Db = np.zeros(number_of_data)
    Di = np.zeros(number_of_data)
    LZC_y = np.zeros(number_of_data)
    P_u = np.zeros(number_of_data)
    P_o = np.zeros(number_of_data)
    P_y = np.zeros(number_of_data)
    P_k = np.zeros(number_of_data)
    P_x = np.zeros(number_of_data)
    P_U = np.zeros(number_of_data)
    P_O = np.zeros(number_of_data)
    P_Y = np.zeros(number_of_data)
    P_K = np.zeros(number_of_data)
    P_X = np.zeros(number_of_data)

    if queue_Fea_Ext_Cal_progress is not None:
        queue_Fea_Ext_Cal_progress.put(("消息类型：特征提取计算进度条 — 初始化", task_index_Fea_Ext_Cal, number_of_data))

    for i in tqdm(range(number_of_data), desc="特征提取"):
        y = raw_data[:, i]
        bw = estimate_main_bandwidth(y, fs)
        norm_freq = min(bw / (fs / 2), 0.99)
        fir_len = len(y)
        b = firwin(fir_len + 1, norm_freq)
        y_filtered = lfilter(b, 1, y)

        # Hilbert 包络
        y_H = np.imag(hilbert(np.real(y_filtered)))
        y_s = np.real(y_filtered) + 1j * y_H
        y_envelope = np.abs(y_s)
        y_envelope_mean[i] = np.mean(y_envelope)

        # RJ 特征
        m2 = np.mean(y_envelope ** 2)
        m4 = np.mean(y_envelope ** 4)
        R_HT[i] = abs((m4 - m2 ** 2) / m2 ** 2)
        J_HT[i] = abs(m4 - 2 * m2 ** 2)

        # 盒维数
        d = 1 / len(y_envelope)
        sum_Db = ((np.maximum(y_envelope[:-1], y_envelope[1:]) - np.minimum(y_envelope[:-1],
                                                                            y_envelope[1:])) * d) / d ** 2
        N_d = len(y_envelope) + np.sum(sum_Db)
        Db[i] = -np.log(N_d) / np.log(d)

        # 信息维数
        y_diff = np.abs(np.diff(y_envelope))
        p_diff = y_diff / np.sum(y_diff)
        p_diff[p_diff == 0] = 1  # 避免 log(0)
        Di[i] = -np.sum(p_diff * np.log10(p_diff))

        # LZC复杂度
        y_dc = y_envelope - np.mean(y_envelope)
        diff = np.abs(np.diff(y_dc))
        thresh = np.mean(diff)
        binary = (diff >= thresh).astype(int)
        y_q_str = "".join(map(str, binary))
        c = 1
        S = y_q_str[0]
        Q = ""
        for ch in y_q_str[1:]:
            Q += ch
            if S.find(Q) == -1:
                S += Q
                Q = ""
                c += 1
        LZC_y[i] = c * np.log10(len(y_q_str)) / len(y_q_str)

        # 信噪比
        q = np.mean(np.abs(y) ** 2)
        m = np.mean(np.abs(y) ** 4)
        snr_val = np.sqrt(2 * q ** 2 - m)
        SNRE[i] = 10 * np.log10(snr_val / (q - snr_val)) if q > snr_val else 0

        # 相位噪声（简化计算）
        w = boxcar(len(y))
        h = np.correlate(w, w, mode="full") / len(y)
        r = np.correlate(y, y, mode="full") / len(y)
        ph_est = np.sum(h * r[:len(h)])
        ph[i] = ph_est

        # 信号特征
        Y = y_filtered - y
        P_u[i] = np.mean(np.abs(Y))
        Y_mean = np.mean(Y)
        P_o[i] = np.mean((np.abs(Y - Y_mean)) ** 2)
        P_y[i] = np.mean((np.abs(Y - Y_mean)) ** 3) / (np.mean((np.abs(Y - Y_mean)) ** 3) ** (1.5))
        P_k[i] = np.mean((np.abs(Y - Y_mean)) ** 4) / (np.mean((np.abs(Y - Y_mean)) ** 2) ** 2)
        y_x1, y_x2 = Y[:len(Y) // 2], Y[len(Y) // 2:]
        P_x[i] = np.mean(np.abs(y_x1)) / np.mean(np.abs(y_x2))

        # 功率谱特征
        P_NOW = fft(y_filtered)
        P_USE = fft(y)
        P_diff = P_NOW - P_USE
        P_U[i] = np.mean(np.abs(P_diff))
        P_mean = np.mean(P_diff)
        P_O[i] = np.mean((np.abs(P_diff - P_mean)) ** 2)
        P_Y[i] = np.mean((np.abs(P_diff - P_mean)) ** 3) / (np.mean((np.abs(P_diff - P_mean)) ** 3) ** (1.5))
        P_K[i] = np.mean((np.abs(P_diff - P_mean)) ** 4) / (np.mean((np.abs(P_diff - P_mean)) ** 2) ** 2)
        P_x1, P_x2 = P_diff[:len(P_diff) // 2], P_diff[len(P_diff) // 2:]
        P_X[i] = np.mean(np.abs(P_x1)) / np.mean(np.abs(P_x2))

        if queue_Fea_Ext_Cal_progress is not None:
            queue_Fea_Ext_Cal_progress.put(("消息类型：特征提取计算进度条 — 更新", task_index_Fea_Ext_Cal, 1))

    # 输出特征矩阵（共18维）
    feature_matrix = np.column_stack([
        np.abs(SNRE), np.abs(ph), y_envelope_mean,
        R_HT, J_HT, Db, Di, LZC_y,
        P_u, P_o, P_y, P_k, P_x,
        P_U, P_O, P_Y, P_K, P_X,
    ])

    logging.info(f"特征提取完成，形状为：{feature_matrix.shape}")

    if queue_Fea_Ext_Cal_progress is not None:
        queue_Fea_Ext_Cal_progress.put(("消息类型：特征提取计算进度条 — 任务结束", task_index_Fea_Ext_Cal, None))

    return feature_matrix