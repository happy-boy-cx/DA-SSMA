import numpy as np
from sklearn.model_selection import train_test_split
import random

def get_num_class_pretraindata():
    x = np.load(f"../Dataset/X_train_90Class.npy")
    y = np.load(f"../Dataset/Y_train_90Class.npy")
    x = x.transpose(0, 2, 1)
    train_index_shot = []
    for i in range(90):
        index_classi = [index for index, value in enumerate(y) if value == i]
        train_index_shot += index_classi[0:100]
    return x[train_index_shot], y[train_index_shot]

def get_num_class_finetunedata(k):
    X_train_label, Y_train_label = TrainDatasetx(10,300)
    X_train_label = X_train_label.transpose(0, 2, 1)
    x_test = np.load(f"../Dataset/run1/x_test_2ft.npy")
    y_test = np.load(f"../Dataset/run1/y_test_2ft.npy")
    x_test = x_test.transpose(0, 2, 1)
    x_test = x_test[:, :4800, :]
    x_test = x_test.transpose(0, 2, 1)

    return X_train_label, x_test, Y_train_label, y_test

def PreTrainDataset_prepared():
    X_train_ul, Y_train_ul = get_num_class_pretraindata()
    Y_train_ul = Y_train_ul.astype(np.uint8)

    min_value = X_train_ul.min()
    max_value = X_train_ul.max()

    X_train_ul = (X_train_ul - min_value) / (max_value - min_value)

    return X_train_ul, Y_train_ul

def FineTuneDataset_prepared(k):
    X_train, X_test, Y_train, Y_test = get_num_class_finetunedata(k)
    Y_train = Y_train.astype(np.uint8)
    Y_test = Y_test.astype(np.uint8)

    max_value = X_train.max()
    min_value = X_train.min()

    X_train = (X_train - min_value) / (max_value - min_value)
    X_test = (X_test - min_value) / (max_value - min_value)

    return X_train, X_test, Y_train, Y_test


def TrainDataset(num, rand_num):
    x = np.load(f"../Dataset/X_train_{num}Class.npy")
    y = np.load(f"../Dataset/Y_train_{num}Class.npy")
    X_train_labeled1, X_train_unlabeled1, Y_train_labeled1, Y_train_unlabeled1 = train_test_split(x, y, test_size=0.5, random_state=rand_num)
    X_train_labeled2, X_train_unlabeled2, Y_train_labeled2, Y_train_unlabeled2 = train_test_split(X_train_labeled1,Y_train_labeled1, test_size=0.2, random_state=rand_num)
    X_train_labeled3, X_train_unlabeled3, Y_train_labeled3, Y_train_unlabeled3 = train_test_split(X_train_labeled2,Y_train_labeled2,test_size=0.5,random_state=rand_num)

    X_train_label, X_val, Y_train_label, Y_val = train_test_split(X_train_labeled3, Y_train_labeled3, test_size=0.2, random_state=rand_num)

    X_train_unlabeled = np.concatenate((X_train_unlabeled1,X_train_unlabeled2,X_train_unlabeled3), axis=0)
    Y_train_unlabeled = np.concatenate((Y_train_unlabeled1, Y_train_unlabeled2,Y_train_unlabeled3), axis=0)


    return X_train_label, x, X_val, Y_train_label, y, Y_val

def TrainDatasetx(num, rand_num):
    x = np.load(f"../Dataset/run1/x_train_2ft.npy")
    y = np.load(f"../Dataset/run1/y_train_2ft.npy")
    x = x.transpose(0, 2, 1)
    x = x[:, :4800, :]
    X_train_labeled1, X_train_unlabeled1, Y_train_labeled1, Y_train_unlabeled1 = train_test_split(x, y, test_size=0.5, random_state=rand_num)
    X_train_labeled2, X_train_unlabeled2, Y_train_labeled2, Y_train_unlabeled2 = train_test_split(X_train_labeled1,Y_train_labeled1, test_size=0.2, random_state=rand_num)
    X_train_labeled3, X_train_unlabeled3, Y_train_labeled3, Y_train_unlabeled3 = train_test_split(X_train_labeled2,Y_train_labeled2,test_size=0.5,random_state=rand_num)

    X_train_label, X_val, Y_train_label, Y_val = train_test_split(X_train_labeled3, Y_train_labeled3, test_size=0.2, random_state=rand_num)

    X_train_unlabeled = np.concatenate((X_train_unlabeled1,X_train_unlabeled2,X_train_unlabeled3), axis=0)
    Y_train_unlabeled = np.concatenate((Y_train_unlabeled1, Y_train_unlabeled2,Y_train_unlabeled3), axis=0)


    return X_train_label, Y_train_label

def TestDataset(num):
    x = np.load(f"../Dataset/X_test_{num}Class.npy")
    y = np.load(f"../Dataset/Y_test_{num}Class.npy")
    y = y.astype(np.uint8)
    return x, y

def FineTuneDataset_prepared_snr(k):
    X_train, X_test, Y_train, Y_test = get_num_class_finetunedata(k)
    Y_train = Y_train.astype(np.uint8)
    Y_test = Y_test.astype(np.uint8)

    max_value = X_train.max()
    min_value = X_train.min()

    X_train = (X_train - min_value) / (max_value - min_value)
    X_test = (X_test - min_value) / (max_value - min_value)

    return X_train, X_test, Y_train, Y_test

def get_num_class_finetunedatax(k,snr):
    X_train_label, Y_train_label = TrainDatasetx(10,300)
    X_train_label = X_train_label.transpose(0, 2, 1)
    x_test = np.load(f"../Dataset/X_test_10Class.npy")
    y_test = np.load(f"../Dataset/Y_test_10Class.npy")
    x_test = add_noise(x_test, snr)
    x_test = x_test.transpose(0, 2, 1)
    return X_train_label, x_test, Y_train_label, y_test

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

X_train_label, x_test, Y_train_label, y_test=get_num_class_finetunedata(5)
a=0
