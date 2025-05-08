import torch
import torch.nn as nn
import math


class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([m], dtype=torch.float32))  # 简化参数定义
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        return fea1 * mix_factor + fea2 * (1 - mix_factor)  # 广播自动处理维度


class Attention(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 保持二维池化兼容性

        # 动态计算卷积核大小
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        # 通道交互层
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, bias=True)  # 等价全连接

        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()

    def forward(self, x):
        # 空间压缩
        attn = self.avg_pool(x)  # [B,C,1,1]

        # 双路径特征提取
        x1 = self.conv1(attn.squeeze(-1).transpose(1, 2)).transpose(1, 2)  # [B,C,1]
        x2 = self.fc(attn).squeeze(-1).transpose(1, 2)  # [B,1,C]

        # 高效矩阵运算
        out1 = torch.einsum('bci,bjc->bc', x1, x2).unsqueeze(-1).unsqueeze(-1)
        out2 = torch.einsum('bci,bjc->bc', x2.transpose(1, 2), x1.transpose(1, 2)).unsqueeze(-1).unsqueeze(-1)

        # 混合注意力
        mixed = self.mix(self.sigmoid(out1), self.sigmoid(out2))

        # 最终注意力调制
        out = self.conv1(mixed.squeeze(-1).transpose(1, 2)).transpose(1, 2).unsqueeze(-1)
        return x * self.sigmoid(out)