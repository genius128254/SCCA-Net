import torch
from torch import nn
import torch.fft



class SpectralAttention(nn.Module):
    def __init__(self, in_dim, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 使用 1x1 卷积替代全连接层
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // reduction_ratio, kernel_size=1, bias=False),  # 压缩通道
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim // reduction_ratio, in_dim, kernel_size=1, bias=False),  # 恢复通道
            nn.Sigmoid()
        )

    def forward(self, x):
        # 全局平均池化
        y = self.avg_pool(x)

        # 通过 1x1 卷积生成注意力权重
        y = self.channel_attention(y)  # 输出形状 [B, C, 1, 1]

        # 对输入进行通道加权
        return x * y


class SSFA(nn.Module):
    def __init__(self, in_dim=224, middle_dim=32, reduced_dim=16, adapt_factor=1):
        super().__init__()
        self.factor = adapt_factor
        self.middle_dim = middle_dim
        self.reduced_dim = reduced_dim

        # 光谱降维（减少通道）
        self.spectral_reduce = nn.Conv2d(in_dim, reduced_dim, 1)
        self.spectral_expand = nn.Conv2d(reduced_dim, in_dim, 1)

        # 空间分支（简化结构）
        self.s_combined_down = nn.Conv2d(reduced_dim, 2 * middle_dim, 1)
        self.s_dw = nn.Conv2d(middle_dim, middle_dim, 3, padding=1, groups=middle_dim)  # 仅保留一个深度卷积
        self.s_relu = nn.ReLU(inplace=True)
        self.s_up = nn.Conv2d(middle_dim, reduced_dim, 1)

        # 频率分支（轻量化）
        self.f_down = nn.Conv2d(reduced_dim, middle_dim, 1)
        self.f_relu1 = nn.ReLU(inplace=True)
        self.f_dw = nn.Conv2d(middle_dim * 2, middle_dim * 2, 1, groups=middle_dim * 2)  # 1x1卷积
        self.f_inter = nn.Sequential(
            nn.Conv2d(middle_dim, middle_dim // 4, 1),  # 缩小中间层
            nn.ReLU(),
            nn.Conv2d(middle_dim // 4, middle_dim, 1)
        )
        self.f_relu2 = nn.ReLU(inplace=True)
        self.f_up = nn.Conv2d(middle_dim, reduced_dim, 1)
        self.sg = nn.Sigmoid()

        # 光谱注意力（更大缩减比例）
        self.spectral_attention = SpectralAttention(reduced_dim)

    def forward(self, x):
        _, _, H, W = x.shape
        x_reduced = self.spectral_reduce(x)

        # 空间分支
        s_combined = self.s_combined_down(x_reduced)
        s1, s2 = torch.chunk(s_combined, 2, dim=1)
        s_product = s1 * s2
        s_dw_out = self.s_dw(s_product)
        s_modulate = self.s_up(self.s_relu(s_dw_out))  # 移除空洞卷积

        # 频率分支
        y = torch.fft.rfft2(self.f_down(x_reduced), norm='backward')
        y_real, y_imag = y.real, y.imag
        y_cat = torch.cat([y_real, y_imag], dim=1)
        y_transformed = self.f_dw(y_cat)  # 1x1卷积
        y_real, y_imag = torch.chunk(y_transformed, 2, dim=1)

        y_amp = torch.sqrt(y_real ** 2 + y_imag ** 2 + 1e-8)
        y_amp_mod = self.f_inter(y_amp)
        y_amp = y_amp * self.sg(y_amp_mod)

        y_real = y_amp * torch.cos(torch.atan2(y_imag, y_real))
        y_imag = y_amp * torch.sin(torch.atan2(y_imag, y_real))
        y = torch.fft.irfft2(torch.complex(y_real, y_imag), s=(H, W), norm='backward')
        f_modulate = self.f_up(self.f_relu2(y))

        # 特征融合
        x_tilde_reduced = x_reduced + (s_modulate + f_modulate) * self.factor
        x_tilde_reduced = self.spectral_attention(x_tilde_reduced)
        x_tilde = self.spectral_expand(x_tilde_reduced)
        return x_tilde




