import torch
from torch import nn
from torch.nn import MaxPool1d, Flatten, BatchNorm1d, LazyLinear, Dropout
from torch.nn import ReLU, Softmax
from complexcnn import ComplexConv
import torch.nn.functional as F

class base_complex_model(nn.Module):
    def __init__(self):
        super(base_complex_model, self).__init__()
        self.conv1 = ComplexConv(sample_len=6000,in_channels=2,out_channels=64,kernel_size=3)
        self.batchnorm1 = BatchNorm1d(num_features=64)
        self.maxpool1 = MaxPool1d(kernel_size=2)

        self.conv2 = ComplexConv(self.conv1.base_out_len,in_channels=64,out_channels=64,kernel_size=3)
        self.batchnorm2 = BatchNorm1d(num_features=64)
        self.maxpool2 = MaxPool1d(kernel_size=2)

        self.conv3 = ComplexConv(self.conv2.base_out_len,in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm3 = BatchNorm1d(num_features=64)
        self.maxpool3 = MaxPool1d(kernel_size=2)

        self.conv4 = ComplexConv(self.conv3.base_out_len,in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm4 = BatchNorm1d(num_features=64)
        self.maxpool4 = MaxPool1d(kernel_size=2)

        self.conv5 = ComplexConv(self.conv4.base_out_len,in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm5 = BatchNorm1d(num_features=64)
        self.maxpool5 = MaxPool1d(kernel_size=2)

        self.conv6 = ComplexConv(self.conv5.base_out_len,in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm6 = BatchNorm1d(num_features=64)
        self.maxpool6 = MaxPool1d(kernel_size=2)

        self.conv7 = ComplexConv(self.conv6.base_out_len,in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm7 = BatchNorm1d(num_features=64)
        self.maxpool7 = MaxPool1d(kernel_size=2)

        self.conv8 = ComplexConv(self.conv7.base_out_len,in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm8 = BatchNorm1d(num_features=64)
        self.maxpool8 = MaxPool1d(kernel_size=2)

        self.conv9 = ComplexConv(self.conv8.base_out_len,in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm9 = BatchNorm1d(num_features=64)
        self.maxpool9 = MaxPool1d(kernel_size=2)

        self.flatten = Flatten()
        self.linear1 = LazyLinear(512)
        self.linear2 = LazyLinear(128)
        self.linear3 = LazyLinear(16)


    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchnorm4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.batchnorm5(x)
        x = self.maxpool5(x)

        x = self.conv6(x)
        x = F.relu(x)
        x = self.batchnorm6(x)
        x = self.maxpool6(x)

        x = self.conv7(x)
        x = F.relu(x)
        x = self.batchnorm7(x)
        x = self.maxpool7(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        embedding = F.relu(x)

        output = self.linear3(embedding)

        return embedding, output

if __name__ == "__main__":
    input = torch.randn((10,2,6000))
    model = base_complex_model()
    embedding, output = model(input)