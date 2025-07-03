import torch
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from get_dataset import *
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sklearn.metrics as sm
from sklearn import manifold
import sys
print(sys.path)
sys.path.insert(0, './models')

def scatter(features, targets, subtitle = None, n_classes = 10):
    palette = np.array(sns.color_palette("hls", n_classes))  # "hls",
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(features[:, 0], features[:, 1], lw=0, s=40, c=palette[targets, :])  #
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    txts = []
    for i in range(n_classes):
        xtext, ytext = np.median(features[targets == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.savefig(f"Visualization/{n_classes}classes_{subtitle}.png", dpi=600)

def visualize_data(data, labels, title, num_clusters):  # feature visualization
    labels = labels.astype(int)
    tsne = manifold.TSNE(n_components=2)  # init='pca'
    data_tsne = tsne.fit_transform(data)
    fig = plt.figure()
    scatter_plot = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], lw=0, s=10, c=labels,
                               cmap=plt.get_cmap("jet", num_clusters))  # 绘制散点图

    # 自定义颜色条标签为1到10
    cbar = plt.colorbar(scatter_plot, ticks=range(num_clusters))  # 设置颜色条的刻度
    cbar.set_ticks(range(num_clusters))  # 设置标签为 0, 1, 2, ..., 9
    cbar.set_ticklabels([str(i + 1) for i in range(num_clusters)])  # 设置标签为 1, 2, 3, ..., 10
    fig.savefig(title, dpi=600)


def Data_prepared(n_classes, rand_num):
    X_train_labeled, X_train, X_val, value_Y_train_labeled, value_Y_train, value_Y_val = TrainDataset(n_classes, rand_num)

    min_value = X_train.min()
    min_in_val = X_val.min()
    if min_in_val < min_value:
        min_value = min_in_val

    max_value = X_train.max()
    max_in_val = X_val.max()
    if max_in_val > max_value:
        max_value = max_in_val

    return max_value, min_value

def TestDataset_prepared(n_classes, rand_num):
    X_test, Y_test = TestDataset(n_classes)

    max_value, min_value = Data_prepared(n_classes, rand_num)

    X_test = (X_test - min_value) / (max_value - min_value)

    X_test = X_test.transpose(0, 2, 1)

    return X_test, Y_test

def obtain_embedding_feature_map(online_network, classifier, test_dataloader, device):
    online_network.eval()
    classifier.eval()
    feature_map = []
    target_output = []
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            # 通过online_network提取特征
            features = online_network(data)[0]  # 假设第一个元素是特征
            # 通过classifier获取预测结果（可选，根据需求）
            # logits = classifier(features)
            # 收集特征和标签
            feature_map.extend(features.cpu().numpy())
            target_output.extend(target.cpu().numpy())
    return np.array(feature_map), np.array(target_output)


def main():
    # 数据预处理
    X_test, Y_test = TestDataset_prepared(10, rand_num=300)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    online_network = torch.load("model_weight/nofrozen_and_onelinear/online_network_pt_0-89_ft_0-9_10shot_22.pth", map_location=device)
    classifier = torch.load("model_weight/nofrozen_and_onelinear/classifier_pt_0-89_ft_0-9_10shot_22.pth", map_location=device)

    # 提取特征
    features, targets = obtain_embedding_feature_map(online_network, classifier, test_dataloader, device)

    # t-SNE可视化
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(features)

    # 绘制图像
    scatter(tsne_results, targets.astype(int), "Combined_Model_Visualization", 10)
    visualize_data(features, targets, "Visualization/Combined_Model_TSNE", 10)

    # 计算轮廓系数
    print("Silhouette Score:", sm.silhouette_score(features, targets))

if __name__ == "__main__":
    main()