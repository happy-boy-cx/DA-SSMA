import os
import sys
print(sys.path)
sys.path.insert(0, './models')
import torch
import yaml
from models.encoder_and_projection import Encoder_and_projection
from models.classifier import Classifier
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from get_dataset import *
import argparse
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



def test(online_network, classifier, test_dataloader, device):
    online_network.eval()  # 启动验证，不允许更新模型参数
    classifier.eval()
    target_pred = []  # 存储所有预测标签
    target_real = []  # 存储所有真实标签
    test_loss = 0
    correct = 0
    loss = nn.NLLLoss()
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
                loss = loss.to(device)
            output = classifier(online_network(data)[0])
            output = F.log_softmax(output, dim=1)
            test_loss += loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            # 收集预测和真实标签
            target_pred.extend(pred.cpu().numpy().flatten())  # 修改收集方式
            target_real.extend(target.cpu().numpy().flatten())  # 修改收集方式

            correct += pred.eq(target.view_as(pred)).sum().item()

            # 转换numpy数组
        target_pred = np.array(target_pred)
        target_real = np.array(target_real)

        # 绘制混淆矩阵
        conf_mat = confusion_matrix(target_real, target_pred)
        classes = np.unique(target_real) + 1  # 自动获取类别标签

        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_mat, annot=True,
                    fmt='d', cmap='Blues',
                    cbar=False,
                    xticklabels=classes,
                    yticklabels=classes)
        plt.xlabel('Predicted label', fontsize=20)
        plt.ylabel('True label', fontsize=20)
        plt.xticks(fontsize=14, rotation=0)
        plt.yticks(fontsize=14, rotation=0)
        plt.title('Confusion Matrix', fontsize=24)
        plt.tight_layout()
        plt.show()

    test_loss /= len(test_dataloader)
    fmt = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )
    return 100.0 * correct / len(test_dataloader.dataset)

def main():
    rand_num = 30
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    config_ft = config['finetune']
    device = torch.device("cuda:0")
    online_network = torch.load("model_weight/nofrozen_and_onelinear/online_network_pt_0-89_ft_0-9_10shot_22.pth",map_location=torch.device('cpu'))
    classifier = torch.load("model_weight/nofrozen_and_onelinear/classifier_pt_0-89_ft_0-9_10shot_22.pth",map_location=torch.device('cpu'))
    X_train, X_test, Y_train, Y_test = FineTuneDataset_prepared_snr(config_ft['k_shot'])
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=config_ft['test_batch_size'], shuffle=True)
    test(online_network, classifier, test_dataloader, device)



if __name__ == "__main__":
    main()