from model_complexcnn import *
import os
from fvcore.nn import FlopCountAnalysis  # 导入FLOPs分析工具

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def calculate_flops(model):
    # 创建一个示例输入，形状应根据你的模型输入进行调整
    dummy_input = torch.randn(1, 2, 4800).to(next(model.parameters()).device)  # 假设输入是3x224x224的图像
    flops = FlopCountAnalysis(model, dummy_input)
    print(f"FLOPs: {flops.total()}")

def calculate_params(model):
    # 计算模型的总参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 将参数数量转换为GB（1个参数假设占用4字节）
    total_params_gb = total_params * 4 / (1024 ** 2)  # 转换为GB

    print(f"Total Params: {total_params} parameters")
    print(f"Total Params (in GB): {total_params_gb:.4f} MB")

def main():
    model = torch.load("model_weight/CNN_DA_SSMA_classes_10_20label_80unlabel_rand30.pth")
    print(model)

    # 计算FLOPs
    calculate_flops(model)

    # 计算模型参数数量
    calculate_params(model)

if __name__ == '__main__':
    main()
