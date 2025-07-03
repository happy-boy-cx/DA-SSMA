import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_complexcnn import *
from get_dataset_10label import *
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from get_dataset_10label import *
from sklearn.linear_model import LogisticRegression
from pandas import DataFrame
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def test(encoder, classifier, test_dataloader):
    encoder.eval()
    classifier.eval()
    test_loss = 0
    correct = 0
    device = torch.device("cuda:0")
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.squeeze().long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            feature = encoder(data)
            logit = classifier(feature)
            output = F.log_softmax(logit, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    fmt = '\nTest set: Accuracy: {}/{} ({:.6f}%)\n'
    print(
        fmt.format(
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )


def Data_prepared(n_classes, rand_num):
    X_train_labeled, X_train_unlabeled, X_train, X_val, value_Y_train_labeled, value_Y_train_unlabeled, value_Y_train, value_Y_val = TrainDataset(n_classes, rand_num)

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
    X_test, value_Y_test = TestDataset(n_classes)

    max_value, min_value = Data_prepared(n_classes, rand_num)

    X_test = (X_test - min_value) / (max_value - min_value)
    X_test = X_test.transpose(0, 2, 1)

    return X_test, value_Y_test

def main():
    rand_num = 30
    encoder = torch.load("model_weight/SimMIM_encoder_mask05_n_classes_10_label10.pth",map_location=torch.device('cpu'))
    classifier = torch.load("model_weight/SimMIM_classifier_mask05_n_classes_10_label10.pth",map_location=torch.device('cpu'))
    X_test, Y_test = TestDataset_prepared(10, rand_num)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 测试时无需shuffle
    test(encoder,classifier,test_dataloader)



if __name__ == '__main__':
   main()