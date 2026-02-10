import numpy as np
from scipy.fftpack import fft, ifft, fftshift
from scipy.signal import hilbert, firwin, lfilter, boxcar
from tqdm import tqdm
from sklearn.preprocessing import normalize

import logging


def ex_feature(original_signal_matrix, Fc, task_index_Fea_Ext_Cal=None, queue_Fea_Ext_Cal_progress=None):
    # # 调试此计算脚本时，用来打印变量的详细值的，平时不用管
    # np.set_printoptions(threshold=np.inf)

    print(f"\n开始特征提取，接收到矩阵形状：{original_signal_matrix.shape}。...")
    logging.info(f"\n开始特征提取，接收到矩阵形状：{original_signal_matrix.shape}。...")

    ''' %% 频域转时域 '''

    # # % 获取输入信号的尺寸，Ndata为数据数量，len1为信号长度
    # number_of_data, length_per_data = original_signal_matrix.shape
    # # % 在信号两侧补零，形成扩展的频域信号
    # Signal_fre = np.hstack(
    #     [np.zeros((number_of_data, 410 + 4 * 12)), original_signal_matrix, np.zeros((number_of_data, 410 + 4 * 12))]
    # )
    # # % 对频域信号进行频移和逆傅里叶变换，转换到时域
    # ''' Signal_time = np.fft.ifft(np.fft.fftshift(Signal_fre.T, axes=0)) 这样算出来和MATLAB对不上'''
    # Signal_time = ifft(fftshift(Signal_fre.T, axes=0), axis=0)
    # # % 获取时域信号的长度
    # len_ = Signal_time.shape[0]
    # # % 储存时域信号数据以备后续处理（注意转置）
    # raw_data = Signal_time  # % 后续的希尔伯特变换以及FFT，是按列进行的，所以进行转置


    raw_data = original_signal_matrix.T
    len_ = raw_data.shape[0]
    Signal_fre = fft(original_signal_matrix, axis=1)
    number_of_data, length_per_data = Signal_fre.shape
    ''' %% 特征值初始化 '''

    # if if_progress_display == 1:
    #     # % 如果处于模式1，则初始化进度条
    #     print("进度条初始化...")

    # % 初始化各种特征值矩阵
    # 星座图
    # theta = np.zeros(number_of_data)
    # Constellation_1 = np.zeros(number_of_data)
    # Constellation_2 = np.zeros(number_of_data)
    # Constellation_3 = np.zeros(number_of_data)

    ''' 注意虚部保留问题 '''
    # 信噪比
    SNRE = np.zeros(number_of_data, dtype=complex)
    # 相位噪声
    ph = np.zeros(number_of_data, dtype=complex)
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

    # % 功率谱

    # % 数据功率归一化
    A_H_1 = np.imag(hilbert(np.real(raw_data), axis=0))  # % 对实数部分进行希尔伯特变换获取虚数部分
    A_s = np.real(raw_data) + 1j * A_H_1  # % 生成解析信号

    print("\n进入范数计算步骤。注意：此步可能触发多线程计算，CPU核心占用数量和占用率快速升高。")
    logging.info("\n进入范数计算步骤。注意：此步可能触发多线程计算，CPU核心占用数量和占用率快速升高。")
    ''' 这一行调了一天，注意 2-范数 和 Frobenius范数 '''
    # A_envelope_HT = np.linalg.norm(A_s)  # % 计算包络的范数
    A_envelope_HT = np.linalg.norm(A_s, 2)  # % 计算包络的范数
    print("\n范数计算步骤结束。")
    logging.info("\n范数计算步骤结束。")

    A_envelope_mean = A_envelope_HT ** 2  # % 计算包络均值平方

    print(
        f"""\n重要参数记录：\nA_envelope_HT: {A_envelope_HT}, A_envelope_mean: {A_envelope_mean}, 
        len(raw_data): {len(raw_data)}, np.sqrt(len(raw_data)): {np.sqrt(len(raw_data))}, 
        raw_data.size: {raw_data.size}, np.sqrt(raw_data.size): {np.sqrt(raw_data.size)}\n"""
    )
    logging.info(
        f"""\n重要参数记录：\nA_envelope_HT: {A_envelope_HT}, A_envelope_mean: {A_envelope_mean}, 
        len(raw_data): {len(raw_data)}, np.sqrt(len(raw_data)): {np.sqrt(len(raw_data))}, 
        raw_data.size: {raw_data.size}, np.sqrt(raw_data.size): {np.sqrt(raw_data.size)}\n"""
    )

    ''' *** 此处注意问题 *** '''
    # raw_data = raw_data * np.sqrt(len(raw_data)) / A_envelope_mean  # % 对原始数据进行归一化
    raw_data = raw_data * np.sqrt(raw_data.size) / A_envelope_mean  # % 对原始数据进行归一化

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
        P_USE = Signal_fre[i, :]  # % 获取对应的频域信号

        # ''' %% 星座图 '''
        #
        # # % 初始化计数器和累加器
        # k, l, m, p = 0, 0, 0, 0
        # y1_sum, y2_sum, y3_sum, y4_sum = 0, 0, 0, 0
        #
        # # % 遍历有效信号部分
        # ''' 注意索引偏移问题 '''
        # # for n in range(459, 459 + length_per_data):
        # for n in range(459 - 1, 459 + length_per_data - 1):
        #     if P_USE[n].real > 0:
        #         if P_USE[n].imag > 0:
        #             k += 1
        #             y1_sum += P_USE[n]  # % 第一象限累加
        #         else:
        #             l += 1
        #             y4_sum += P_USE[n]  # % 第四象限累加
        #     elif P_USE[n].real <= 0:
        #         if P_USE[n].imag > 0:
        #             m += 1
        #             y2_sum += P_USE[n]  # % 第二象限累加
        #         elif P_USE[n].imag < 0:
        #             p += 1
        #             y3_sum += P_USE[n]  # % 第三象限累加
        #
        # # % 计算各象限的均值
        # y1_mean = y1_sum / k if k != 0 else 0
        # y2_mean = y2_sum / m if m != 0 else 0
        # y3_mean = y3_sum / p if p != 0 else 0
        # y4_mean = y4_sum / l if l != 0 else 0
        # # % 计算星座图特征
        # A51 = abs(y1_mean - y2_mean)
        # A61 = abs(y2_mean - y3_mean)
        # A71 = abs(y4_mean - y3_mean)
        # A81 = abs(y4_mean - y1_mean)
        # K1 = abs(y1_mean - y3_mean)
        # K2 = abs(y2_mean - y4_mean)
        # A91 = [A51, A61, A71, A81]
        # theta1 = abs(180 / np.pi * np.angle((y1_mean - y2_mean) / (y4_mean - y1_mean)))
        # theta2 = abs(180 / np.pi * np.angle((y1_mean - y2_mean) / (y2_mean - y3_mean)))
        # theta3 = abs(180 / np.pi * np.angle((y2_mean - y3_mean) / (y3_mean - y4_mean)))
        # theta4 = abs(180 / np.pi * np.angle((y3_mean - y4_mean) / (y4_mean - y1_mean)))
        # theta0 = [theta1, theta2, theta3, theta4]
        #
        # # % 储存特征值
        # theta[i] = 180 - abs(
        #     180 / np.pi * np.angle((y1_mean - y3_mean) / (y2_mean - y3_mean))
        # )
        # Constellation_1[i] = max(A91) / min(A91) if min(A91) != 0 else 0
        # Constellation_2[i] = max(K1, K2) / min(K1, K2) if min(K1, K2) != 0 else 0
        # Constellation_3[i] = max(theta0)

        ''' %% 信噪比 '''

        # % 计算信号的二阶矩
        q = np.sum(y * np.conj(y)) / len(y)  # % 二阶矩
        # % 计算信号的四阶矩
        m = np.sum((np.conj(y) * y) ** 2) / len(y)  # % 四阶矩
        # % 计算信噪比
        SNRE[i] = 10 * np.log10(np.sqrt(2 * q ** 2 - m) / (q - np.sqrt(2 * q ** 2 - m)))
        # % 取绝对值
        SNRE_f = abs(SNRE)

        ''' %% 相位噪声 '''

        ph_n = len_  # % 获取信号长度
        w = boxcar(ph_n)  # % 生成矩形窗
        h = np.correlate(w, w, mode="full") / ph_n  # % 计算自相关
        r = np.correlate(y, y, mode="full") / len(y)  # % 计算信号的自相关
        p_h = np.zeros(2 * len_ - 1, dtype=complex)  # % 初始化相位噪声数组

        # % 计算相位噪声
        ''' 注意错误索引（索引偏移）下面译法是错的 '''
        # for ii in range(-(len_ - 1), len_):
        #     p_h[ii + len_] = (
        #         h[ii + len_] * r[ii + len_] * np.exp(-1j * 2 * np.pi * Fc * ii)
        #     )
        for ii in range(-(len_ - 1), len_):
            p_h[ii + len_ - 1] = (
                    h[ii + len_ - 1] * r[ii + len_ - 1] * np.exp(-1j * 2 * np.pi * Fc * ii)
            )

        ''' 注意虚部保留问题 '''
        ph[i] = np.sum(p_h)  # % 储存相位噪声值
        ph_f = abs(ph)  # % 取绝对值

        ''' %% 信号预处理  基带信号低通滤波 '''

        # N = len_  # % 获取信号长度
        # wc = 3180 / N  # % 计算归一化截止频率
        # ''' b = firwin(N, wc)  # % 设计FIR低通滤波器 这个是错的 '''
        # b = firwin(N + 1, wc)  # % 设计FIR低通滤波器
        # y_after_fir = lfilter(b, 1, y)  # % 对信号进行滤波
        y_after_fir=y

        ''' %% Hilbert变化取包络 '''

        # % 过滤后的信号
        y_1_1 = y_after_fir
        # % 计算Hilbert变换的虚部
        y_H_1 = np.imag(hilbert(np.real(y_1_1)))  # Hilbert变化
        # % 构造解析信号
        y_s = np.real(y_1_1) + 1j * y_H_1  # 解析信号
        # % 计算信号包络
        y_envelope_HT = abs(y_s)  # 包络
        # % 计算包络均值
        y_envelope_mean[i] = np.sum(y_envelope_HT) / len(y_envelope_HT)

        ''' %% 取包络特征 '''

        y_envelope_use = y_envelope_HT

        ''' %% RJ特征 '''

        # % 初始化RJ特征数组
        y_envelope = np.zeros(len(y_envelope_use))
        # % 计算包络的二阶矩
        m2_y_envelope = np.sum(np.abs(y_envelope_use) ** 2) / len(
            y_envelope_use
        )  # 二阶矩
        # % 计算包络的四阶矩
        for m in range(len(y_envelope_use)):
            y_envelope[m] = y_envelope_use[m] ** 4
        # % 计算包络的四阶矩
        m4_y_envelope = np.sum(y_envelope) / len(y_envelope_use)  # 四阶矩

        # % 计算R特征
        R_HT[i] = abs((m4_y_envelope - m2_y_envelope ** 2) / m2_y_envelope ** 2)  # R特征
        # % 计算J特征
        J_HT[i] = abs((m4_y_envelope - 2 * m2_y_envelope ** 2))

        ''' %% 盒维数 '''

        d = 1 / len(y_envelope_use)  # % 计算分箱宽度
        sum_Db = np.zeros(len(y_envelope_use) - 1)  # % 初始化盒维数累加数组
        # % 计算盒维数（关于突然出现这个HT：是一回事）
        for m in range(len(y_envelope_use) - 1):
            sum_Db[m] = (
                                max(y_envelope_use[m], y_envelope_use[m + 1]) * d
                                - min(y_envelope_use[m], y_envelope_HT[m + 1]) * d
                        ) / d ** 2
        N_d = len(y_envelope_use) + np.sum(sum_Db)  # % 计算盒维数总和
        Db[i] = -np.log(N_d) / np.log(d)  # % 计算盒维数

        ''' %% 信息维数 '''

        # % 初始化信息维数数组
        y0 = np.zeros(len(y_envelope_use) - 1)
        # % 计算包络差分
        y_0 = np.abs(np.diff(y_envelope_use))
        # % 归一化差分值
        p_0 = y_0 / np.sum(y_0)
        # % 计算信息熵
        di = p_0 * (np.log10(p_0))
        # % 计算信息维数
        Di[i] = -np.sum(di)  # 信息维数

        ''' %%  LZC复杂度 '''

        # % 去直流分量
        y_a = y_envelope_use - np.sum(y_envelope_use) / len(y_envelope_use)  # 去直流
        # % 初始化差分数组
        y_c = np.zeros(len(y_envelope_use))
        # % 计算差分
        for n in range(len(y_envelope_use) - 1):
            y_c[n] = abs(y_a[n + 1] - y_a[n])
        # % 初始化二值化数组
        y_q = np.zeros(len(y_c) - 1)
        # % 二值化处理
        for n in range(len(y_c) - 1):
            y_q[n] = 0 if y_c[n] < np.sum(y_c) / len(y_c) else 1

        y_q_str = "".join(map(str, y_q.astype(int)))  # % 转换为字符串
        c = 1  # % 初始化复杂度计数器
        S = y_q_str[0]  # % 初始化字符串S
        Q = ""  # % 初始化字符串Q
        SQ = ""  # % 初始化组合字符串SQ

        for n in range(1, len(y_q_str)):  # % 计算LZC复杂度
            Q += y_q_str[n]
            SQ = S + Q
            SQv = SQ[:-1]
            if SQv.find(Q) == -1:
                S = SQ
                Q = ""
                c += 1

                # % 计算LZC复杂度特征值
        LZC_y[i] = c * np.log10(len(y_q_str)) / len(y_q_str)  # LZC特征值

        ''' %% 信号特征 '''

        Y = y_1_1 - y  # % 计算信号差分
        P_u[i] = np.sum(abs(Y)) / len(Y)  # % 计算均值
        Y_mean = np.sum(Y) / len(Y)  # % 计算信号均值
        P_o[i] = np.sum((abs(Y - Y_mean) ** 2)) / len(Y)  # % 计算信号方差
        P_y[i] = (np.sum((abs(Y - Y_mean) ** 3)) / len(Y)) / (  # % 计算偏度
                np.sum((abs(Y - Y_mean) ** 3)) ** 1.5 / len(Y)
        )
        P_k[i] = (np.sum((abs(Y - Y_mean) ** 4)) / len(Y)) / (  # % 计算峰度
                np.sum((abs(Y - Y_mean) ** 2)) ** 2 / len(Y)
        )
        y_x1 = Y[: len(Y) // 2]  # % 前半部分信号
        y_x2 = Y[len(Y) // 2:]  # % 后半部分信号
        ''' y_x2 = Y[len(Y) // 2 + 1:]  # % 后半部分信号 这个写法是错的 '''
        P_x[i] = (np.sum(abs(y_x1)) / len(Y)) / (np.sum(abs(y_x2)) / len(Y))  # % 计算前后半部分信号的比率

        ''' %% 功率谱特征 '''
        P_NOW = fft(y_1_1)  # % 计算当前信号的傅里叶变换
        P = P_NOW - P_USE  # % 计算功率谱差分

        P_U[i] = np.sum(abs(P)) / len(P)  # % 计算总功率
        P_mean = np.sum(P) / len(P)  # % 计算功率均值
        P_O[i] = np.sum((abs(P - P_mean) ** 2)) / len(P)  # % 计算功率方差
        P_Y[i] = (np.sum((abs(P - P_mean) ** 3)) / len(P)) / (  # % 计算功率偏度
                np.sum((abs(P - P_mean) ** 3)) ** 1.5 / len(P)
        )
        P_K[i] = (np.sum((abs(P - P_mean) ** 4)) / len(P)) / (  # % 计算功率峰度
                np.sum((abs(P - P_mean) ** 2)) ** 2 / len(P)
        )
        P_x1 = P[: len(P) // 2]  # % 前半部分功率谱
        P_x2 = P[len(P) // 2:]  # % 后半部分功率谱
        ''' P_x2 = P[len(P) // 2 + 1 :]  # 后半部分功率谱 这个写法是错的 '''
        P_X[i] = (np.sum(abs(P_x1)) / len(P)) / (np.sum(abs(P_x2)) / len(P))  # % 计算前后半部分功率谱的比率

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
            # theta, Constellation_1, Constellation_2, Constellation_3,
            SNRE_f,
            ph,
            y_envelope_mean,
            R_HT, J_HT,
            Db,
            Di,
            LZC_y,
            # P_u, P_o, P_y, P_k, P_x,
            # P_U, P_O, P_Y, P_K, P_X,
        ]
    )
    # % 取特征值的绝对值
    Feature = np.abs(feature_matrix)
    # # % 可选的特征归一化
    # Feature = normalize(Feature, axis=1)

    print(f"结束特征提取，返回的矩阵形状：{Feature.shape}")
    logging.info(f"结束特征提取，返回的矩阵形状：{Feature.shape}")

    if queue_Fea_Ext_Cal_progress is not None:
        queue_Fea_Ext_Cal_progress.put(
            ("消息类型：特征提取计算进度条 — 任务结束", task_index_Fea_Ext_Cal, None)
        )

    return Feature