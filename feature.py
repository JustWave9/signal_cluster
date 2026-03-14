import numpy as np
from scipy.fftpack import fft, ifft, fftshift
from scipy.signal import hilbert, firwin, lfilter, boxcar, welch, detrend
from typing import Optional, Tuple, Dict
from tqdm import tqdm
from sklearn.preprocessing import normalize

import logging


def _median_positive(x: np.ndarray) -> float:
    """Median of positive values; fallback to median of all; 0 if empty."""
    x = np.asarray(x)
    xp = x[x > 0]
    if xp.size > 0:
        return float(np.median(xp))
    return float(np.median(x)) if x.size > 0 else 0.0


def phase_increment_series_no_ref(
        y: np.ndarray,
        amp_thr_ratio: float = 0.08,
        min_valid_ratio: float = 0.20,
        unwrap: bool = True,
) -> np.ndarray:
    """
    无参考：提取相位增量序列
        Δφ[n] = angle( y[n] * conj(y[n-1]) )
    - 幅度门限屏蔽 GI=0 / 弱信号点
    - 只保留相邻点都有效的位置
    返回 dphi (rad)，若有效点不足则返回空数组。
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a 1-D complex array")
    if y.size < 3:
        return np.array([], dtype=float)

    amp = np.abs(y)
    med = _median_positive(amp)
    if med <= 0:
        return np.array([], dtype=float)

    thr = amp_thr_ratio * med
    valid = amp > thr

    # 相邻点都有效才算
    v = valid[1:] & valid[:-1]
    if float(np.mean(v)) < min_valid_ratio:
        return np.array([], dtype=float)

    u = y / (np.abs(y)+ 1e-12)
    z = u[1:][v] * np.conj(u[:-1][v])

    # 避免 angle(接近0) 抖动
    z_amp = np.abs(z)
    z_med = _median_positive(z_amp)
    z_thr = 1e-12 + 0.05 * z_med
    z = z[z_amp > z_thr]
    if z.size < 32:
        return np.array([], dtype=float)

    dphi = np.angle(z)  # (-pi, pi]
    if unwrap:
        dphi = np.unwrap(dphi)
    return dphi


def phase_noise_psd_no_ref(
        y: np.ndarray,
        fs: float = 1e7,
        nperseg: int = 2048,
        noverlap: Optional[int] = None,
        amp_thr_ratio: float = 0.08,
        detrend_linear: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    无参考近似相位噪声 PSD（对 Δφ 序列）：
    - 提取 dphi
    - 去均值（去 CFO 的主要项）
    - 可选去线性趋势
    - Welch 得到 Sdphi(f) [rad^2/Hz]
    返回 (f, Sdphi, dphi_used)；失败返回 (None, None, None)
    """
    if fs <= 0:
        raise ValueError("fs must be positive")

    dphi = phase_increment_series_no_ref(y, amp_thr_ratio=amp_thr_ratio)
    if dphi.size < 128:
        return None, None, None

    # 去平均频偏（CFO）
    dphi = dphi - np.mean(dphi)

    if detrend_linear:
        dphi = detrend(dphi, type="linear")

    nperseg_eff = int(min(nperseg, dphi.size))
    if nperseg_eff < 128:
        return None, None, None

    if noverlap is None:
        noverlap_eff = nperseg_eff // 2
    else:
        noverlap_eff = int(min(noverlap, nperseg_eff - 1))

    f, Sdphi = welch(
        dphi,
        fs=fs,
        window="hann",
        nperseg=nperseg_eff,
        noverlap=noverlap_eff,
        detrend=False,
        scaling="density",
        return_onesided=True,
    )
    return f, Sdphi, dphi


def _bandpower(f: np.ndarray, S: np.ndarray, f1: float, f2: float) -> float:
    """∫ S(f) df over [f1,f2] using trapz. Returns NaN if band not available."""
    if f is None or S is None:
        return float("nan")
    if f1 >= f2:
        return float("nan")
    mask = (f >= f1) & (f <= f2)
    if not np.any(mask):
        return float("nan")
    return float(np.trapz(S[mask], f[mask]))


def phase_noise_features_no_ref(
        y: np.ndarray,
        fs: float = 1e7,
        nperseg: int = 2048,
        amp_thr_ratio: float = 0.08,
        detrend_linear: bool = False,
        # len_=10000 => 1 ms => Δf ~ 1kHz，故建议使用 kHz~MHz 频段
        bands_hz: Tuple[Tuple[float, float], ...] = ((2e3, 3e4), (2e3, 1e4), (1e4, 3e4), (3e4, 8e4)),
        return_spectrum: bool = False,
) -> Dict[str, float]:
    """
    输出适合聚类的无参考相位噪声特征：
    - dphi_rms (rad)
    - band_* (rad^2) : 各频段积分
    - total_band (rad^2) : 从 f[1] 到 Nyquist 的积分（避开 f=0）
    可选返回谱 f, Sdphi（return_spectrum=True）
    """
    f, Sdphi, dphi = phase_noise_psd_no_ref(
        y, fs=fs, nperseg=nperseg, amp_thr_ratio=amp_thr_ratio,
        detrend_linear=detrend_linear
    )

    if dphi is None or dphi.size == 0:
        dphi_rms = float("nan")
    else:
        dphi_rms = float(np.sqrt(np.mean(dphi ** 2)))

    out: Dict[str, float] = {"dphi_rms": dphi_rms}

    for (a, b) in bands_hz:
        key = "band_{}_{}".format(int(a), int(b))
        out[key] = _bandpower(f, Sdphi, a, b)

    if f is not None and Sdphi is not None and f.size >= 2:
        out["total_band"] = float(np.trapz(Sdphi[1:], f[1:]))
    else:
        out["total_band"] = float("nan")

    if return_spectrum:
        # 这两个键给你画图/调试用
        out["__f_len__"] = float(0 if f is None else len(f))
        out["__Sdphi_mean__"] = float("nan" if Sdphi is None else np.mean(Sdphi))

    return out

def ex_feature(original_signal_matrix, Fs, task_index_Fea_Ext_Cal=None, queue_Fea_Ext_Cal_progress=None):
    # # 调试此计算脚本时，用来打印变量的详细值的，平时不用管
    # np.set_printoptions(threshold=np.inf)

    print(f"\n开始特征提取，接收到矩阵形状：{original_signal_matrix.shape}。...")
    logging.info(f"\n开始特征提取，接收到矩阵形状：{original_signal_matrix.shape}。...")

    ''' %% 频域转时域 '''
    Signal_fre = fft(original_signal_matrix, axis=1)
    number_of_data, length_per_data = Signal_fre.shape
    raw_data = original_signal_matrix.T
    len_ = raw_data.shape[0]

    ''' %% 特征值初始化 '''


    # % 初始化各种特征值矩阵

    ''' 注意虚部保留问题 '''
    # 信噪比
    SNRE = np.zeros(number_of_data, dtype=complex)
    #IQ  不圆度
    rho = np.zeros(number_of_data)
    c20_re_n = np.zeros(number_of_data)
    c20_im_n = np.zeros(number_of_data)
    dphi_std = np.zeros(number_of_data)
    image_rej_db = np.zeros(number_of_data)
    #mean(y**2)相位
    psi_cos= np.zeros(number_of_data)
    psi_sin=np.zeros(number_of_data)
    # 相位噪声
    ph = np.zeros(number_of_data, dtype=complex)
    pn_b1 = np.zeros(number_of_data)  # 5k~50k
    pn_b2 = np.zeros(number_of_data)  # 50k~500k
    pn_b3 = np.zeros(number_of_data)  # 500k~2M
    # 包络
    y_envelope_mean = np.zeros(number_of_data)
    # RJ特征
    R_HT = np.zeros(number_of_data)
    J_HT = np.zeros(number_of_data)
    # 盒维数
    Db = np.zeros(number_of_data)
    # 信息维数
    Di = np.zeros(number_of_data)
    # LZC复杂度
    LZC_y = np.zeros(number_of_data)
    # 信号特征
    P_u = np.zeros(number_of_data)
    P_o = np.zeros(number_of_data)
    P_y = np.zeros(number_of_data)
    P_k = np.zeros(number_of_data)
    P_x = np.zeros(number_of_data)
    # 功率谱特征
    P_U = np.zeros(number_of_data)
    P_O = np.zeros(number_of_data)
    P_Y = np.zeros(number_of_data)
    P_K = np.zeros(number_of_data)
    P_X = np.zeros(number_of_data)

    ''' %% 特征提取 '''
    eps=1e-12
    log_eps=1e-20

    # A_s=raw_data
    # envelope_mean=np.abs(A_s)
    # phase = np.angle(A_s)
    #
    # p = np.mean(np.abs(raw_data) ** 2, axis=0)  # (K,) 每段平均功率
    # # raw_data = raw_data / np.sqrt(p)


    # 向主进程发送信号条数，用于在主进程中初始化该任务的进度条
    if queue_Fea_Ext_Cal_progress is not None:
        # queue_Fea_Ext_Cal_progress.put(
        #     ("消息类型：特征提取计算进度条 — 循环开始")
        # )
        queue_Fea_Ext_Cal_progress.put(
            ("消息类型：特征提取计算进度条 — 初始化", task_index_Fea_Ext_Cal, len(original_signal_matrix))
        )

    for i in tqdm(range(number_of_data), desc="当前矩阵提取进度"):  # % 遍历每个数据样本

        y = raw_data[:, i]  # % 获取第i个样本的时域数据
        # P_USE = Signal_fre[i, :]  # % 获取对应的频域信号

        ''' %% 信噪比 '''
        q = np.mean(np.abs(y) ** 2)     #  二阶矩
        m = np.mean(np.abs(y) ** 4)     #     四阶矩
        tmp = 2 * q ** 2 - m
        tmp = max(tmp, eps)
        sqrt_term = np.sqrt(tmp)

        den = q - sqrt_term
        den = max(den, eps)

        ratio = sqrt_term / den
        ratio = max(ratio, eps)

        SNRE[i] = 10 * np.log10(ratio)


        ''' IQ  不圆度'''
        y0=y-np.mean(y)
        c20 = np.mean(y0 ** 2)
        den_rho = np.mean(np.abs(y0) ** 2)
        den_rho = max(den_rho, eps)

        rho[i] = np.abs(c20) / den_rho
        c20_re_n[i] = np.real(c20) / den_rho
        c20_im_n[i] = np.imag(c20) / den_rho

        c20_abs = max(np.abs(c20), eps)
        psi_cos[i] = np.real(c20) / c20_abs
        psi_sin[i] = np.imag(c20) / c20_abs

        # ===== 新增 2：dphi_std =====
        dphi = phase_increment_series_no_ref(
            y,
            amp_thr_ratio=0.08,
            min_valid_ratio=0.20,
            unwrap=True
        )

        if dphi.size < 8:
            dphi_std[i] = 0.0
        else:
            # 去掉均值，避免把平均频偏当成波动
            dphi_centered = dphi - np.mean(dphi)
            dphi_std[i] = np.std(dphi_centered)

        # ===== 新增 3：image_rej_db =====
        # 估计带内镜像抑制度：正负频带能量比
        nfft_img = int(2 ** np.ceil(np.log2(len(y))))
        Y = np.fft.fftshift(np.fft.fft(y, n=nfft_img))
        f = np.fft.fftshift(np.fft.fftfreq(nfft_img, d=1.0 / Fs))

        # 用你的信号主带宽来截取，避免吃太多带外噪声
        # 若你已知 BW_99_max≈83kHz，可先写死；后续也可改成参数
        bw_use = 8.0e4

        pos_mask = (f > 0) & (f < bw_use)
        neg_mask = (f < 0) & (f > -bw_use)

        P_pos = np.sum(np.abs(Y[pos_mask]) ** 2)
        P_neg = np.sum(np.abs(Y[neg_mask]) ** 2)

        P_pos = max(P_pos, eps)
        P_neg = max(P_neg, eps)

        # 镜像抑制度，越大通常说明镜像越小
        image_rej_db[i] = 10.0 * np.log10(P_pos / P_neg)


        ''' %% 相位噪声 '''
        pn = phase_noise_features_no_ref(y, Fs)
        # 你原来 ph[i] 是一个标量：建议先用 total_band 替代
        ph[i] = np.log10(pn["band_2000_30000"]+ 1e-20)
        pn_b1[i] = np.log10(pn["band_2000_10000"]+ 1e-20)
        pn_b2[i] = np.log10(pn["band_10000_30000"]+ 1e-20)
        pn_b3[i] = np.log10(pn["band_30000_80000"]+ 1e-20)

        ''' %% 取包络特征 RJ特征 '''
        y_envelope_use = np.abs(y)
        y_envelope_mean[i] = np.mean(y_envelope_use)

        m2_y_envelope = np.mean(y_envelope_use ** 2)
        m4_y_envelope = np.mean(y_envelope_use ** 4)

        den_rj = max(m2_y_envelope ** 2, eps)

        R_HT[i] = abs((m4_y_envelope - m2_y_envelope ** 2) / den_rj)
        J_HT[i] = abs(m4_y_envelope - 2 * m2_y_envelope ** 2)

        ''' %% 盒维数 '''

        d = 1 / len(y_envelope_use)  # % 计算分箱宽度
        sum_Db = np.zeros(len(y_envelope_use) - 1)  # % 初始化盒维数累加数组
        # % 计算盒维数（关于突然出现这个HT：是一回事）
        for m in range(len(y_envelope_use) - 1):
            sum_Db[m] = (
                                max(y_envelope_use[m], y_envelope_use[m + 1]) * d
                                - min(y_envelope_use[m], y_envelope_use[m + 1]) * d
                        ) / (d ** 2)
        N_d = len(y_envelope_use) + np.sum(sum_Db)  # % 计算盒维数总和
        N_d = max(N_d,eps)
        Db[i] = -np.log(N_d) / np.log(max(d,eps))  # % 计算盒维数

        ''' %% 信息维数 '''

        # % 计算包络差分
        y_0 = np.abs(np.diff(y_envelope_use))
        sum_y0 = np.sum(y_0)
        if sum_y0 < eps:
            Di[i] = 0.0
        else:
            p_0 = y_0 / sum_y0
            p_0 = np.maximum(p_0, eps)
            di = p_0 * np.log10(p_0)
            Di[i] = -np.sum(di)

        ''' %%  LZC复杂度 '''

        # % 去直流分量
        y_a = y_envelope_use - np.sum(y_envelope_use) / len(y_envelope_use)  # 去直流
        # % 计算差分
        y_c = np.abs(np.diff(y_a))

        if len(y_c) < 2 or np.mean(y_c) < eps:
            LZC_y[i] = 0.0
            continue
        #  二值化阈值（更稳健：median；也可用 mean）
        else:
            y_q = (y_c >= (np.mean(y_c))).astype(np.uint8)  # 0/1 序列
            y_q_str = "".join(map(str, y_q.astype(int)))  # % 转换为字符串

            c = 1  # % 初始化复杂度计数器
            S = y_q_str[0]  # % 初始化字符串S
            Q = ""  # % 初始化字符串Q

            for n in range(1, len(y_q_str)):  # % 计算LZC复杂度
                Q += y_q_str[n]
                SQ = S + Q
                SQv = SQ[:-1]
                if SQv.find(Q) == -1:
                    S = SQ
                    Q = ""
                    c += 1

            if Q != "":
                c += 1
                    # % 计算LZC复杂度特征值
            LZC_y[i] = c * np.log10(len(y_q_str)) / len(y_q_str)  # LZC特征值


        ''' %% 信号特征 '''
        #
        # Y = y_1_1 - y  # % 计算信号差分
        # P_u[i] = np.sum(abs(Y)) / len(Y)  # % 计算均值
        # Y_mean = np.sum(Y) / len(Y)  # % 计算信号均值
        # P_o[i] = np.sum((abs(Y - Y_mean) ** 2)) / len(Y)  # % 计算信号方差
        # P_y[i] = (np.sum((abs(Y - Y_mean) ** 3)) / len(Y)) / (  # % 计算偏度
        #         np.sum((abs(Y - Y_mean) ** 3)) ** 1.5 / len(Y)
        # )
        # P_k[i] = (np.sum((abs(Y - Y_mean) ** 4)) / len(Y)) / (  # % 计算峰度
        #         np.sum((abs(Y - Y_mean) ** 2)) ** 2 / len(Y)
        # )
        # y_x1 = Y[: len(Y) // 2]  # % 前半部分信号
        # y_x2 = Y[len(Y) // 2:]  # % 后半部分信号
        # ''' y_x2 = Y[len(Y) // 2 + 1:]  # % 后半部分信号 这个写法是错的 '''
        # P_x[i] = (np.sum(abs(y_x1)) / len(Y)) / (np.sum(abs(y_x2)) / len(Y))  # % 计算前后半部分信号的比率
        #
        # ''' %% 功率谱特征 '''
        # P_NOW = fft(y_1_1)  # % 计算当前信号的傅里叶变换
        # P = P_NOW - P_USE  # % 计算功率谱差分
        #
        # P_U[i] = np.sum(abs(P)) / len(P)  # % 计算总功率
        # P_mean = np.sum(P) / len(P)  # % 计算功率均值
        # P_O[i] = np.sum((abs(P - P_mean) ** 2)) / len(P)  # % 计算功率方差
        # P_Y[i] = (np.sum((abs(P - P_mean) ** 3)) / len(P)) / (  # % 计算功率偏度
        #         np.sum((abs(P - P_mean) ** 3)) ** 1.5 / len(P)
        # )
        # P_K[i] = (np.sum((abs(P - P_mean) ** 4)) / len(P)) / (  # % 计算功率峰度
        #         np.sum((abs(P - P_mean) ** 2)) ** 2 / len(P)
        # )
        # P_x1 = P[: len(P) // 2]  # % 前半部分功率谱
        # P_x2 = P[len(P) // 2:]  # % 后半部分功率谱
        # ''' P_x2 = P[len(P) // 2 + 1 :]  # 后半部分功率谱 这个写法是错的 '''
        # P_X[i] = (np.sum(abs(P_x1)) / len(P)) / (np.sum(abs(P_x2)) / len(P))  # % 计算前后半部分功率谱的比率

        # if if_progress_display == 1:
        #     # % 更新进度条
        #     print(f"射频指纹提取进度: {round((i + 1) / number_of_data * 100)}%, 第{i + 1}个样本")

        if queue_Fea_Ext_Cal_progress is not None:
            queue_Fea_Ext_Cal_progress.put(
                ("消息类型：特征提取计算进度条 — 更新", task_index_Fea_Ext_Cal, 1)
            )

    ''' %% 整合特征 '''

    # % 汇总所有特征值
    feature_matrix = np.column_stack(
        [
            SNRE,
            # psi_cos,
            # psi_sin,
            # rho,
            # c20_re_n,
            # c20_im_n,
            # dphi_std,
            # image_rej_db,
            # ph,
            # pn_b1,
            # pn_b2,
            # pn_b3,
            y_envelope_mean,
            R_HT,
            J_HT,
            # Db,
            # Di,
            # LZC_y,
            # P_u, P_o, P_y, P_k, P_x,
            # P_U, P_O, P_Y, P_K, P_X,
        ]
    )
    # % 取特征值的绝对值
    Feature = np.abs(feature_matrix)
    # # % 可选的特征归一化
    # Feature = normalize(Feature, axis=1)

    # print(f"结束特征提取，返回的矩阵形状：{Feature.shape}")
    # logging.info(f"结束特征提取，返回的矩阵形状：{Feature.shape}")

    if queue_Fea_Ext_Cal_progress is not None:
        queue_Fea_Ext_Cal_progress.put(
            ("消息类型：特征提取计算进度条 — 任务结束", task_index_Fea_Ext_Cal, None)
        )

    return Feature
    # return feature_matrix


