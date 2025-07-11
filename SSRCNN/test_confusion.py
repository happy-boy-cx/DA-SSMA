import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from get_dataset_10label import *
from SSRCNN_Complex import *
from sklearn.linear_model import LogisticRegression
from pandas import DataFrame
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
import seaborn as sns

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def test(model, test_dataloader):
    model.eval()
    correct = 0
    device = torch.device("cuda:0")
    target_pred = []
    target_real = []
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = model(data)
            classifier_value = F.log_softmax(output[1], dim=1)
            pred = classifier_value.argmax(dim=1, keepdim=True)
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
        plt.tight_layout()
        plt.show()

    fmt = '\nTest set: Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )

def Data_prepared(n_classes, rand_num):
    X_train_label, X_train_unlabel, X_train, X_val, Y_train_label, Y_train_unlabel, Y_train, Y_val = TrainDatasetx(n_classes, rand_num)

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

def main():
    rand_num = 30
    X_test, Y_test = TestDataset_prepared(10, rand_num)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    model = torch.load("model_weight/SSRCNN_n_classes_10_10label_90unlabel_rand30.pth",map_location=torch.device('cpu'))
    test(model,test_dataloader)

if __name__ == '__main__':
   main()
