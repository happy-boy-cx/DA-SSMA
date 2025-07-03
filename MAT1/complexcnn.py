# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

from MAT1.AFFNet import DAConv


class ComplexConv(nn.Module):
    def __init__(self, sample_len, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(ComplexConv, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding

        # 原复数卷积组件（实部和虚部分支）
        self.conv_re = nn.Conv1d(
            in_channels=in_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.conv_im = nn.Conv1d(
            in_channels=in_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        # 后置DAConv模块（方案一）
        self.daconv = DAConv(
            sample_len=sample_len,
            in_channels=out_channels,  # 输入通道=复数卷积的输出通道
            out_channels=out_channels,  # 输出通道保持相同
            kernel_size=3,
            stride=1,
            padding=1,
            spatial_att_enable=True,
            channel_att_enable=True,
            residual_enable=True
        )
        self.base_out_len =int((self.daconv.out_len+2*1-(3-1)-1)/2 + 1)
        a=0

    def forward(self, x):
        # 分离实部和虚部
        x_real = x[:, 0:x.shape[1] // 2, :]
        x_img = x[:, x.shape[1] // 2:, :]

        # 复数卷积运算
        real = self.conv_re(x_real) - self.conv_im(x_img)
        imaginary = self.conv_re(x_img) + self.conv_im(x_real)
        output = torch.cat((real, imaginary), dim=1)

        # 动态更新DAConv的输入长度
        current_length = output.shape[-1]
        self.daconv.sample_len = current_length  # 传递给DAConv
        self.daconv.out_len = current_length  # 确保DAConv内部使用最新长度

        output = self.daconv(output)
        return output
