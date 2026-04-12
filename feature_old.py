import numpy as np
from scipy.signal import welch, detrend
from typing import Optional, Tuple, Dict
from tqdm import tqdm
import logging


def _median_positive(x: np.ndarray) -> float:
    """Median of positive values; fallback to median of all; 0 if empty."""
    x = np.asarray(x)
    xp = x[x > 0]
    if xp.size > 0:
        return float(np.median(xp))
    return float(np.median(x)) if x.size > 0 else 0.0


def _bandpower(f: np.ndarray, S: np.ndarray, f1: float, f2: float) -> float:
    """对 [f1, f2] 上的 PSD 做积分。无有效频点时返回 NaN。"""
    if f is None or S is None:
        return float("nan")
    if f1 >= f2:
        return float("nan")
    mask = (f >= f1) & (f <= f2)
    if not np.any(mask):
        return float("nan")
    return float(np.trapz(S[mask], f[mask]))


def phase_increment_series_no_ref(
        y: np.ndarray,
        amp_thr_ratio: float = 0.08,
        min_valid_ratio: float = 0.20,
) -> np.ndarray:
    """
    无参考相位增量序列:
        Δφ[n] = angle( y[n] * conj(y[n-1]) )

    说明：
    - 仅保留相邻点都有效的位置
    - 用幅度门限屏蔽 GI=0 / 弱信号点
    - 不做 unwrap，保留局部相位增量主值
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

    u = y / (np.abs(y) + 1e-12)
    z = u[1:][v] * np.conj(u[:-1][v])

    # 避免 angle(接近0) 抖动
    z_amp = np.abs(z)
    z_med = _median_positive(z_amp)
    z_thr = 1e-12 + 0.05 * z_med
    z = z[z_amp > z_thr]
    if z.size < 32:
        return np.array([], dtype=float)

    dphi = np.angle(z)  # (-pi, pi]
    dphi = np.unwrap(dphi)
    return dphi


def phase_increment_abs_rms_feature(
        y: np.ndarray,
        amp_thr_ratio: float = 0.08,
        min_valid_ratio: float = 0.20,
) -> float:
    """
    相位增量波动特征（你最终定下来的版本）：
        dphi_rms = sqrt(mean((|Δφ|-mean(|Δφ|))^2))

    含义：
    - 强调相位变化幅值是否稳定
    - 弱化 MSK 理想 ± 方向切换带来的影响
    """
    dphi = phase_increment_series_no_ref(
        y,
        amp_thr_ratio=amp_thr_ratio,
        min_valid_ratio=min_valid_ratio
    )
    if dphi.size < 8:
        return 0.0

    adphi = np.abs(dphi)
    mu = np.mean(adphi)
    return float(np.sqrt(np.mean((adphi - mu) ** 2)))

def phase_increment_rms_feature(
        y: np.ndarray,
        amp_thr_ratio: float = 0.08,
        min_valid_ratio: float = 0.20,
) -> float:
    """
    相位增量波动特征（不加绝对值版本）：
        dphi_rms = sqrt(mean((Δφ-mean(Δφ))^2))

    含义：
    - 直接刻画原始相位增量围绕其均值的波动程度
    - 保留相位增量的正负方向信息
    """
    dphi = phase_increment_series_no_ref(
        y,
        amp_thr_ratio=amp_thr_ratio,
        min_valid_ratio=min_valid_ratio
    )
    if dphi.size < 8:
        return 0.0

    mu = np.mean(dphi)
    return float(np.sqrt(np.mean((dphi - mu) ** 2)))

def phase_noise_psd_no_ref(
        y: np.ndarray,
        fs: float = 1e7,
        nperseg: int = 2048,
        noverlap: Optional[int] = None,
        amp_thr_ratio: float = 0.08,
        detrend_linear: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    基于原始相位增量 Δφ 的 Welch PSD：
    - 提取 dphi
    - 去均值
    - 可选去线性趋势
    - Welch 得到 Sdphi(f)
    """
    if fs <= 0:
        raise ValueError("fs must be positive")

    dphi = phase_increment_series_no_ref(y, amp_thr_ratio=amp_thr_ratio)
    if dphi.size < 128:
        return None, None, None

    # 谱特征保留原始相位增量版本，只做去均值
    dphi_used = dphi - np.mean(dphi)

    if detrend_linear:
        dphi_used = detrend(dphi_used, type="linear")

    nperseg_eff = int(min(nperseg, dphi_used.size))
    if nperseg_eff < 128:
        return None, None, None

    if noverlap is None:
        noverlap_eff = nperseg_eff // 2
    else:
        noverlap_eff = int(min(noverlap, nperseg_eff - 1))

    f, Sdphi = welch(
        dphi_used,
        fs=fs,
        window="hann",
        nperseg=nperseg_eff,
        noverlap=noverlap_eff,
        detrend=False,
        scaling="density",
        return_onesided=True,
    )
    return f, Sdphi, dphi_used


def phase_noise_features_no_ref(
        y: np.ndarray,
        fs: float = 1e7,
        nperseg: int = 2048,
        amp_thr_ratio: float = 0.08,
        detrend_linear: bool = False,
        in_band_hz: Tuple[float, float] = (2e3, 5e4),
        out_band_hz: Tuple[float, float] = (5e4, 8e4),
        return_spectrum: bool = False,
) -> Dict[str, float]:
    """
    相位增量谱特征：
    - P_in:  固定带内积分
    - P_out: 固定带外积分

    所有信号统一使用相同积分范围，便于横向比较。
    """
    f, Sdphi, dphi = phase_noise_psd_no_ref(
        y,
        fs=fs,
        nperseg=nperseg,
        amp_thr_ratio=amp_thr_ratio,
        detrend_linear=detrend_linear
    )

    out: Dict[str, float] = {}

    if f is None or Sdphi is None or len(f) < 2:
        out["P_in"] = float("nan")
        out["P_out"] = float("nan")
        return out

    f_in_1, f_in_2 = in_band_hz
    f_out_1, f_out_2 = out_band_hz

    out["P_in"] = _bandpower(f, Sdphi, f_in_1, f_in_2)
    out["P_out"] = _bandpower(f, Sdphi, f_out_1, f_out_2)

    if return_spectrum:
        out["__f_len__"] = float(len(f))
        out["__Sdphi_mean__"] = float(np.mean(Sdphi))

    return out


def ex_feature(
        original_signal_matrix: np.ndarray,
        Fs: float,
        task_index_Fea_Ext_Cal=None,
        queue_Fea_Ext_Cal_progress=None,
        in_band_hz: Tuple[float, float] = (2e3, 5e4),
        out_band_hz: Tuple[float, float] = (5e4, 8e4),
        force_nonnegative: bool = False,
) -> np.ndarray:
    """
    提取 13 维手工特征：
    [SNRE, rho, c20_re_n, c20_im_n, dphi_std, P_in, P_out,
     y_envelope_mean, R_HT, J_HT, Db, Di, LZC_y]

    参数
    ----
    original_signal_matrix : shape = (N, L)
        每行一个样本（复数基带序列）
    Fs : float
        采样率
    in_band_hz : (f1, f2)
        相位增量谱带内积分范围
    out_band_hz : (f1, f2)
        相位增量谱带外积分范围
    force_nonnegative : bool
        若为 True，则按列平移到非负区间。
        注意：这是“平移”，不是取绝对值，不会抹掉 c20_re_n/c20_im_n 的方向信息。
    """
    print(f"\n开始特征提取，接收到矩阵形状：{original_signal_matrix.shape}。...")
    logging.info(f"\n开始特征提取，接收到矩阵形状：{original_signal_matrix.shape}。...")

    original_signal_matrix = np.asarray(original_signal_matrix)
    if original_signal_matrix.ndim != 2:
        raise ValueError("original_signal_matrix must be a 2-D array, shape=(N, L)")

    number_of_data, length_per_data = original_signal_matrix.shape
    raw_data = original_signal_matrix.T   # shape=(L, N)

    # ===== 特征初始化 =====
    eps = 1e-12

    # 信噪比
    SNRE = np.zeros(number_of_data, dtype=float)

    # IQ 不圆度
    rho = np.zeros(number_of_data, dtype=float)
    c20_re_n = np.zeros(number_of_data, dtype=float)
    c20_im_n = np.zeros(number_of_data, dtype=float)

    # 相位扰动
    dphi_std = np.zeros(number_of_data, dtype=float)   # 这里保留变量名，含义已变成“绝对值波动特征”
    P_in = np.zeros(number_of_data, dtype=float)
    P_out = np.zeros(number_of_data, dtype=float)

    # 包络
    y_envelope_mean = np.zeros(number_of_data, dtype=float)

    # RJ 特征
    R_HT = np.zeros(number_of_data, dtype=float)
    J_HT = np.zeros(number_of_data, dtype=float)

    # 盒维数 / 信息维数 / LZC
    Db = np.zeros(number_of_data, dtype=float)
    Di = np.zeros(number_of_data, dtype=float)
    LZC_y = np.zeros(number_of_data, dtype=float)

    # 初始化进度条
    if queue_Fea_Ext_Cal_progress is not None:
        queue_Fea_Ext_Cal_progress.put(
            ("消息类型：特征提取计算进度条 — 初始化", task_index_Fea_Ext_Cal, len(original_signal_matrix))
        )

    for i in tqdm(range(number_of_data), desc="当前矩阵提取进度"):
        y = raw_data[:, i]

        # ===== 1) SNR 估计 =====
        q = np.mean(np.abs(y) ** 2)   # 二阶矩
        m = np.mean(np.abs(y) ** 4)   # 四阶矩

        tmp = max(2 * q ** 2 - m, eps)
        sqrt_term = np.sqrt(tmp)

        den = max(q - sqrt_term, eps)
        ratio = max(sqrt_term / den, eps)

        SNRE[i] = 10.0 * np.log10(ratio)

        # ===== 2) IQ 非圆性特征 =====
        y0 = y - np.mean(y)
        c20 = np.mean(y0 ** 2)

        den_rho = max(np.mean(np.abs(y0) ** 2), eps)
        rho[i] = np.abs(c20) / den_rho
        c20_re_n[i] = np.real(c20) / den_rho
        c20_im_n[i] = np.imag(c20) / den_rho

        # ===== 3) 相位增量波动特征（非绝对值版本） =====
        dphi_std[i] = phase_increment_rms_feature(
            y,
            amp_thr_ratio=0.08,
            min_valid_ratio=0.20
        )

        # ===== 4) 相位增量谱特征（原始相位增量版本） =====
        pn = phase_noise_features_no_ref(
            y,
            fs=Fs,
            nperseg=2048,
            amp_thr_ratio=0.08,
            detrend_linear=False,
            in_band_hz=in_band_hz,
            out_band_hz=out_band_hz
        )

        P_in[i] = 0.0 if np.isnan(pn["P_in"]) else pn["P_in"]
        P_out[i] = 0.0 if np.isnan(pn["P_out"]) else pn["P_out"]

        # ===== 5) 包络 + RJ =====
        y_envelope_use = np.abs(y)
        y_envelope_mean[i] = np.mean(y_envelope_use)

        m2_y_envelope = np.mean(y_envelope_use ** 2)
        m4_y_envelope = np.mean(y_envelope_use ** 4)
        den_rj = max(m2_y_envelope ** 2, eps)

        R_HT[i] = abs((m4_y_envelope - m2_y_envelope ** 2) / den_rj)
        J_HT[i] = abs(m4_y_envelope - 2 * m2_y_envelope ** 2)

        # ===== 6) 盒维数 =====
        d = 1.0 / len(y_envelope_use)
        sum_Db = np.zeros(len(y_envelope_use) - 1)

        for m_idx in range(len(y_envelope_use) - 1):
            sum_Db[m_idx] = (
                max(y_envelope_use[m_idx], y_envelope_use[m_idx + 1]) * d
                - min(y_envelope_use[m_idx], y_envelope_use[m_idx + 1]) * d
            ) / (d ** 2)

        N_d = max(len(y_envelope_use) + np.sum(sum_Db), eps)
        Db[i] = -np.log(N_d) / np.log(max(d, eps))

        # ===== 7) 信息维数 =====
        y_0 = np.abs(np.diff(y_envelope_use))
        sum_y0 = np.sum(y_0)

        if sum_y0 < eps:
            Di[i] = 0.0
        else:
            p_0 = y_0 / sum_y0
            p_0 = np.maximum(p_0, eps)
            di = p_0 * np.log10(p_0)
            Di[i] = -np.sum(di)

        # ===== 8) LZC 复杂度 =====
        y_a = y_envelope_use - np.mean(y_envelope_use)
        y_c = np.abs(np.diff(y_a))

        if len(y_c) < 2 or np.mean(y_c) < eps:
            LZC_y[i] = 0.0
        else:
            y_q = (y_c >= np.mean(y_c)).astype(np.uint8)
            y_q_str = "".join(map(str, y_q.astype(int)))

            c = 1
            S = y_q_str[0]
            Q = ""

            for n in range(1, len(y_q_str)):
                Q += y_q_str[n]
                SQ = S + Q
                SQv = SQ[:-1]
                if SQv.find(Q) == -1:
                    S = SQ
                    Q = ""
                    c += 1

            if Q != "":
                c += 1

            LZC_y[i] = c * np.log10(len(y_q_str)) / len(y_q_str)

        if queue_Fea_Ext_Cal_progress is not None:
            queue_Fea_Ext_Cal_progress.put(
                ("消息类型：特征提取计算进度条 — 更新", task_index_Fea_Ext_Cal, 1)
            )

    # ===== 整合特征 =====
    feature_matrix = np.column_stack(
        [
            SNRE,
            rho,
            c20_re_n,
            c20_im_n,
            dphi_std,
            P_in,
            P_out,
            y_envelope_mean,
            R_HT,
            J_HT,
            Db,
            Di,
            LZC_y,
        ]
    ).astype(np.float64)

    # 清理 NaN / inf
    # Feature = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    Feature = np.abs(feature_matrix)
    # 如果你后续代码坚持要求非负，就按列平移，不要取绝对值
    if force_nonnegative:
        col_min = Feature.min(axis=0)
        Feature = Feature - col_min + 1e-12

    if queue_Fea_Ext_Cal_progress is not None:
        queue_Fea_Ext_Cal_progress.put(
            ("消息类型：特征提取计算进度条 — 任务结束", task_index_Fea_Ext_Cal, None)
        )

    return Feature