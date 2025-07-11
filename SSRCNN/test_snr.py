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
            correct += pred.eq(target.view_as(pred)).sum().item()
            target_pred[len(target_pred):len(target)-1] = pred.tolist()
            target_real[len(target_real):len(target)-1] = target.tolist()

        target_pred = np.array(target_pred)
        target_real = np.array(target_real)

        print(precision_score(target_real, target_pred, average='macro'))
        print(recall_score(target_real, target_pred, average='macro'))
        print(f1_score(target_real, target_pred, average='macro'))

    # # 将预测标签存下来
    # data_Y_pred = pd.DataFrame(target_pred)
    # writer = pd.ExcelWriter("SSRCNN_15label/Y_pred.xlsx")
    # data_Y_pred.to_excel(writer, 'page_1', float_format='%.5f')
    # writer.save()
    # writer.close()
    #
    # # 将原始标签存下来
    #
    # data_Y_real = pd.DataFrame(target_real)
    # writer = pd.ExcelWriter("SSRCNN_15label/Y_real.xlsx")
    # data_Y_real.to_excel(writer, 'page_1', float_format='%.5f')
    # writer.save()
    # writer.close()

    fmt = '\nTest set: Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )
    accuracy = 100.0 * correct / len(test_dataloader.dataset)
    return accuracy

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

def TestDataset_prepared(n_classes, rand_num,snr_db):
    X_test, Y_test = TestDatasetx(n_classes,snr_db)

    max_value, min_value = Data_prepared(n_classes, rand_num)

    X_test = (X_test - min_value) / (max_value - min_value)

    X_test = X_test.transpose(0, 2, 1)

    return X_test, Y_test

def main():
    rand_num = 30
    snr_list = [-10, -5, 0, 5, 10, 15]
    accuracies = []
    model = torch.load("model_weight/SSRCNN_n_classes_10_10label_90unlabel_rand30.pth",map_location=torch.device('cpu'))
    # 测试不同SNR
    for snr_db in snr_list:
        X_test, Y_test = TestDataset_prepared(10, rand_num, snr_db)
        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 测试时无需shuffle
        accuracy = test(model, test_dataloader)
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

if __name__ == '__main__':
   main()
