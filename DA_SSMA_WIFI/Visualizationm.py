import torch
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from get_dataset_10label import *
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.metrics import silhouette_score


def scatter(features, targets, subtitle=None, n_classes=16):
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    # 离散颜色分配：前5类为红色系，后5类为蓝色系
    color_list = [
        '#010080', '#0100CC', '#0008FF', '#004DFF', '#0090FF', '#00D5FF', '#2AFFCF', '#60FF98',  # 红-橙-黄
        '#97FF60', '#CEFF29', '#FEE600', '#FFA600', '#FF6800', '#FE2900', '#CC0001', '#800000'  # 蓝-青
    ]
    cmap_custom = mcolors.ListedColormap(color_list)

    # 将离散类别映射到连续色阶
    bounds = np.linspace(0, n_classes, n_classes + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap_custom.N)
    sc = ax.scatter(features[:, 0], features[:, 1],
                    c=targets,
                    cmap=cmap_custom,
                    s=40,
                    edgecolors='none',
                    vmin=0,
                    vmax=n_classes - 1)

    # 设置坐标范围（参考图像范围）
    ax.set_xlim(-70, 70)
    ax.set_ylim(-70, 70)

    # 颜色条设置
    cbar = plt.colorbar(sc, ax=ax,
                        ticks=np.arange(n_classes),  # 标签居中
                        spacing='proportional')
    cbar.set_ticklabels([str(i) for i in range(1,17)])
    cbar.ax.tick_params(labelsize=10)

    # 边框设置
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(f"Visualization/{n_classes}classes_{subtitle}.png",
                dpi=600,
                bbox_inches='tight')
    # Show the plot
    plt.show()
    plt.close()


def Data_prepared(n_classes):
    # 保持原有数据预处理逻辑
    X_train_labeled, X_train_unlabeled, X_train, X_val, value_Y_train_labeled, value_Y_train_unlabeled, value_Y_train, value_Y_val = TrainDatasetx(
        n_classes, 300)

    min_value = min(X_train.min(), X_val.min())
    max_value = max(X_train.max(), X_val.max())

    return max_value, min_value


def TestDataset_prepared(n_classes):
    X_test, Y_test = TestDataset(n_classes)
    max_value, min_value = Data_prepared(n_classes)

    # 添加数值稳定性保护
    eps = 1e-8
    X_test = (X_test - min_value) / (max_value - min_value + eps)

    # 调整输入形状适配模型（根据实际需求调整维度顺序）
    X_test = X_test.transpose(0, 2, 1)  # 改为 (samples, channels, time)
    return X_test, Y_test


def obtain_embedding_feature_map(model, test_dataloader):
    model.eval()
    feature_map = []
    target_output = []
    with torch.no_grad():
        for data, target in test_dataloader:
            data = data.float()  # 确保数据类型正确
            output = model(data)
            # 假设主输出是元组的第一个元素
            main_output = output[0]  # ✅ 提取元组中的第一个张量
            feature_map.extend(main_output.cpu().numpy().tolist())  # ✅ 转为CPU上的numpy
            target_output.extend(target.numpy().tolist())
    return np.array(feature_map), np.array(target_output)


def main():
    # 加载数据
    X_test, Y_test = TestDataset_prepared(10)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 关闭shuffle保证顺序

    # 加载模型到CPU
    # model = torch.load('model_weight/CNN_DA_SSMA_classes_10label_90unlabel_rand30.pth',
    #                    map_location=torch.device('cpu'))
    model = torch.load('model_weight/CNN_DA_SSMA_classes_10label_90unlabel_rand30.pth',
                       map_location=torch.device('cpu'))
    model.float()  # 确保模型参数为Float类型

    # 提取特征
    X_test_embedding_feature_map, target = obtain_embedding_feature_map(model, test_dataloader)

    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=30)  # 调整perplexity优化效果
    eval_tsne_embeds = tsne.fit_transform(X_test_embedding_feature_map)

    # 可视化
    scatter(eval_tsne_embeds, target.astype(int), "CNN_DA_SSMA_16classes_CPU", 16)

    # 计算轮廓系数
    silhouette_avg = silhouette_score(eval_tsne_embeds, target.astype(int), metric='euclidean')
    print(f"轮廓系数: {silhouette_avg}")


if __name__ == "__main__":
    main()