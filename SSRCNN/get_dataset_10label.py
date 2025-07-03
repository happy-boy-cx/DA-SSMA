import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random

def TrainDataset(num,rand_num):
    x = np.load(f"../Dataset/X_train_{num}Class.npy")
    y = np.load(f"../Dataset/Y_train_{num}Class.npy")
    y = y.astype(np.uint8)
    X_train_labeled1, X_train_unlabeled1, Y_train_labeled1, Y_train_unlabeled1 = train_test_split(x, y, test_size=0.5, random_state=rand_num)
    X_train_labeled2, X_train_unlabeled2, Y_train_labeled2, Y_train_unlabeled2 = train_test_split(X_train_labeled1,Y_train_labeled1, test_size=0.6, random_state=rand_num)
    X_train_labeled3, X_train_unlabeled3, Y_train_labeled3, Y_train_unlabeled3 = train_test_split(X_train_labeled2,Y_train_labeled2,test_size=0.5,random_state=rand_num)

    X_train_label, X_val, Y_train_label, Y_val = train_test_split(X_train_labeled3, Y_train_labeled3, test_size=0.3, random_state=rand_num)

    X_train_unlabeled = np.concatenate((X_train_unlabeled1,X_train_unlabeled2,X_train_unlabeled3), axis=0)
    Y_train_unlabeled = np.concatenate((Y_train_unlabeled1, Y_train_unlabeled2,Y_train_unlabeled3), axis=0)

    return X_train_label, x, X_val, Y_train_label, y, Y_val

def TrainDatasetx(num,rand_num):
    x = np.load(f"../Dataset/X_train_{num}Class.npy")
    y = np.load(f"../Dataset/Y_train_{num}Class.npy")
    y = y.astype(np.uint8)
    X_train_labeled1, X_train_unlabeled1, Y_train_labeled1, Y_train_unlabeled1 = train_test_split(x, y, test_size=0.5, random_state=rand_num)
    X_train_labeled2, X_train_unlabeled2, Y_train_labeled2, Y_train_unlabeled2 = train_test_split(X_train_labeled1,Y_train_labeled1, test_size=0.6, random_state=rand_num)
    X_train_labeled3, X_train_unlabeled3, Y_train_labeled3, Y_train_unlabeled3 = train_test_split(X_train_labeled2,Y_train_labeled2,test_size=0.5,random_state=rand_num)

    X_train_label, X_val, Y_train_label, Y_val = train_test_split(X_train_labeled3, Y_train_labeled3, test_size=0.3, random_state=rand_num)

    X_train_unlabeled = np.concatenate((X_train_unlabeled1,X_train_unlabeled2,X_train_unlabeled3), axis=0)
    Y_train_unlabeled = np.concatenate((Y_train_unlabeled1, Y_train_unlabeled2,Y_train_unlabeled3), axis=0)

    return X_train_label,X_train_unlabeled, x, X_val, Y_train_label,Y_train_unlabeled, y, Y_val

def TestDataset(num):
    x = np.load(f"../Dataset/X_test_{num}Class.npy")
    y = np.load(f"../Dataset/Y_test_{num}Class.npy")
    y = y.astype(np.uint8)
    return x, y

def TestDatasetx(num,snr_db):
    x = np.load(f"../Dataset/X_test_{num}Class.npy")
    y = np.load(f"../Dataset/Y_test_{num}Class.npy")
    y = y.astype(np.uint8)
    x_noisy = add_noise(x, snr_db)
    return x_noisy, y

def add_noise(x, snr_db):
    """
    给信号添加高斯噪声以模拟给定的SNR值。

    参数：
        x: 输入信号（numpy数组）
        snr_db: 给定的SNR值（单位：dB）

    返回：
        带噪声的信号
    """
    # 计算信号的功率
    signal_power = np.mean(np.abs(x) ** 2)

    # 将SNR从dB转换为线性值
    snr_linear = 10 ** (snr_db / 10)

    # 计算噪声的功率
    noise_power = signal_power / snr_linear

    # 生成与信号相同形状的高斯噪声
    noise = np.sqrt(noise_power) * np.random.randn(*x.shape)

    # 将噪声添加到信号
    x_noisy = x + noise
    return x_noisy
