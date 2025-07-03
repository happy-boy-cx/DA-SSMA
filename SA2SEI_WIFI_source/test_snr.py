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

def test(online_network, classifier, test_dataloader, device):
    online_network.eval()  # 启动验证，不允许更新模型参数
    classifier.eval()
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
            correct += pred.eq(target.view_as(pred)).sum().item()

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
    snr_list = [-10, -5, 0, 5, 10, 15]
    accuracies = []
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    config_ft = config['finetune']
    device = torch.device("cuda:0")
    online_network = torch.load("model_weight/nofrozen_and_onelinear/online_network_pt_0-89_ft_0-9_10shot_22.pth",map_location=torch.device('cpu'))
    classifier = torch.load("model_weight/nofrozen_and_onelinear/classifier_pt_0-89_ft_0-9_10shot_22.pth",map_location=torch.device('cpu'))
    # 测试不同SNR
    for snr_db in snr_list:
        X_train, X_test, Y_train, Y_test = FineTuneDataset_prepared_snr(config_ft['k_shot'],snr_db)
        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
        test_dataloader = DataLoader(test_dataset, batch_size=config_ft['test_batch_size'], shuffle=True)
        accuracy = test(online_network, classifier, test_dataloader, device)
        accuracies.append(accuracy)
        print(f"SNR={snr_db}dB, Accuracy={accuracy:.2f}%")

    # 保存结果到CSV
    df = pd.DataFrame({"SNR(dB)": snr_list, "Accuracy(%)": accuracies})
    df.to_csv("accuracy_vs_snr.csv", index=False)

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(snr_list, accuracies, 'b^-', linewidth=2, markersize=8)
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Classification Accuracy under Different SNR", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(snr_list)
    # 设置 y 轴的刻度从 0 开始，每 10 一个刻度
    plt.yticks(range(0, 100 + 10, 10))
    plt.savefig("snr_accuracy_curve.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()