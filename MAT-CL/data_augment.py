import numpy as np
import random


def phase_rotation(data, angle_deg):
    """相位旋转增强 (保持原始数据维度)"""
    angle_rad = np.deg2rad(angle_deg)

    # 正确分离I/Q通道 → (batch_size, seq_len)
    i = data[:, :, 0]  # 原代码错误点：data[:, 0, :] → 错误维度
    q = data[:, :, 1]  # 原代码错误点：data[:, 1, :] → 错误维度

    # 应用旋转矩阵
    i_rot = i * np.cos(angle_rad) - q * np.sin(angle_rad)
    q_rot = i * np.sin(angle_rad) + q * np.cos(angle_rad)

    # 合并到通道维度 → axis=2
    return np.stack([i_rot, q_rot], axis=2)


def random_flip(data, flip_type):
    flipped = data.copy()

    if flip_type == 'horizontal' or flip_type == 'both':
        # 时间序列翻转
        flipped = flipped[:, :, ::-1]
    if flip_type == 'vertical' or flip_type == 'both':
        # 信号幅度翻转
        flipped *= -1

    return flipped


def add_noise(data, snr_db):
    """根据SNR添加高斯噪声 (保持数据范围)"""
    # 计算信号功率
    signal_power = np.mean(data ** 2)

    # 根据SNR计算噪声标准差
    noise_std = np.sqrt(signal_power / (10 ** (snr_db / 10)))

    # 生成高斯噪声
    noise = np.random.normal(0, noise_std, data.shape)

    return data + noise


def apply_augmentations(X_labeled, Y_labeled, X_unlabeled, Y_unlabeled, num_augments=4):
    # 标注数据增强（强增强）
    labeled_aug_X = [X_labeled]
    labeled_aug_Y = [Y_labeled]
    angles = random.sample([30, 60, 90, 180, 270], k=num_augments)  #
    for angle in angles:
        flip_mode = random.choice(['horizontal', 'vertical', 'both'])  # 强制翻转
        snr = np.random.randint(15, 20)  # 低信噪比
        x = phase_rotation(X_labeled, angle)
        x = random_flip(x, flip_mode)
        x = add_noise(x, snr)

        labeled_aug_X.append(x)
        labeled_aug_Y.append(Y_labeled)

    # 未标注数据增强（弱增强）
    unlabeled_aug_X = [X_unlabeled]
    unlabeled_aug_Y = [Y_unlabeled]

    anglex=[10,20,30,40]
    flip_modex = ['horizontal','vertical','horizontal',None]  #
    i=0
    for angle in anglex:
        flip_mode = flip_modex[i]
        snr = np.random.randint(25, 30)  # 高信噪比
        x = phase_rotation(X_unlabeled, angle)
        x = random_flip(x, flip_mode)
        x = add_noise(x, snr)
        unlabeled_aug_X.append(x)
        unlabeled_aug_Y.append(Y_unlabeled)
        i=i+1

    # 将增强数据转换为ndarray类型
    labeled_aug_X = np.array(labeled_aug_X)
    labeled_aug_Y = np.array(labeled_aug_Y)
    unlabeled_aug_X = np.array(unlabeled_aug_X)
    unlabeled_aug_Y = np.array(unlabeled_aug_Y)

    # 将原始数据与增强数据合并
    augmented_X = np.concatenate(labeled_aug_X, axis=0)  # 按批次合并
    augmented_Y = np.concatenate(labeled_aug_Y, axis=0)  # 按批次合并

    augmented_X_unlabeled = np.concatenate(unlabeled_aug_X, axis=0)
    augmented_Y_unlabeled = np.concatenate(unlabeled_aug_Y, axis=0)

    # 返回增强后的数据
    return augmented_X, augmented_Y, augmented_X_unlabeled, augmented_Y_unlabeled

def apply_augmentationsx(X_labeled, Y_labeled, X_unlabeled, Y_unlabeled, num_augments=2):
    # 标注数据增强（强增强）
    labeled_aug_X = [X_labeled]
    labeled_aug_Y = [Y_labeled]
    angles = random.sample([30, 60, 90, 180, 270], k=num_augments)  #
    for angle in angles:
        flip_mode = random.choice(['horizontal', 'vertical', 'both'])  # 强制翻转
        snr = np.random.randint(25, 30)  # 低信噪比
        x = phase_rotation(X_labeled, angle)
        x = random_flip(x, flip_mode)
        x = add_noise(x, snr)

        labeled_aug_X.append(x)
        labeled_aug_Y.append(Y_labeled)

    # 未标注数据增强（弱增强）
    unlabeled_aug_X = [X_unlabeled]
    unlabeled_aug_Y = [Y_unlabeled]

    anglex=[10,40]
    flip_modex = ['horizontal','vertical','horizontal',None]  #
    i=0
    for angle in anglex:
        flip_mode = flip_modex[i]
        snr = np.random.randint(10, 15)  # 高信噪比
        x = phase_rotation(X_unlabeled, angle)
        x = random_flip(x, flip_mode)
        x = add_noise(x, snr)
        unlabeled_aug_X.append(x)
        unlabeled_aug_Y.append(Y_unlabeled)
        i=i+1

    # 将增强数据转换为ndarray类型
    labeled_aug_X = np.array(labeled_aug_X)
    labeled_aug_Y = np.array(labeled_aug_Y)
    unlabeled_aug_X = np.array(unlabeled_aug_X)
    unlabeled_aug_Y = np.array(unlabeled_aug_Y)

    # 将原始数据与增强数据合并
    augmented_X = np.concatenate(labeled_aug_X, axis=0)  # 按批次合并
    augmented_Y = np.concatenate(labeled_aug_Y, axis=0)  # 按批次合并

    augmented_X_unlabeled = np.concatenate(unlabeled_aug_X, axis=0)
    augmented_Y_unlabeled = np.concatenate(unlabeled_aug_Y, axis=0)

    # 返回增强后的数据
    return augmented_X, augmented_Y, augmented_X_unlabeled, augmented_Y_unlabeled