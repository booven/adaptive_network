import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureAttention(nn.Module):
    def __init__(self, in_channels):
        super(TextureAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.pa = SpatialAttention()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fea = self.conv(x)
        att_map = self.pa(fea)

        return att_map

"""
Channel Attention and Spaitial Attention from    
Woo, S., Park, J., Lee, J.Y., & Kweon, I. CBAM: Convolutional Block Attention Module. ECCV2018.
"""


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class TemporalAttention(nn.Module):
    def __init__(self, rnn_size: int):
        super(TemporalAttention, self).__init__()
        self.w = nn.Linear(rnn_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, H):
        # eq.9: M = tanh(H)
        M = self.tanh(H)  # (batch_size, word_pad_len, rnn_size)

        # eq.10: α = softmax(w^T M)
        alpha = self.w(M).squeeze(2)  # (batch_size, word_pad_len)
        alpha = self.softmax(alpha)  # (batch_size, word_pad_len)

        # eq.11: r = H
        r = H * alpha.unsqueeze(2)  # (batch_size, word_pad_len, rnn_size)
        r = r.sum(dim=1)  # (batch_size, rnn_size)

        return r, alpha


"""
参考自FcaNet: "Frequency Channel Attention Networks": https://github.com/cfzd/FcaNet
在线阅读："https://openaccess.thecvf.com/content/ICCV2021/papers/Qin_FcaNet_Frequency_Channel_Attention_Networks_ICCV_2021_paper.pdf"
"""

# 获取频率分量对应的索引，AdfNet采用"FcaNet: Frequency Channel Attention Networks"论文中实验得出效果最佳的结果
# 作为初始频域分量，并加上可学习的频域分量以此自适应学习有效噪声残差

def get_freq_indices(method):
    mapper_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2]
    mapper_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2]
    return mapper_x, mapper_y


class AFCA(nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16):
        super(AFCA, self).__init__()
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices()
        self.num_split = len(mapper_x)  # 多少个频率分量分成多少个部分，一个特征图用一个频率分量预处理
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        y = self.dct_layer(x_pooled)
        y = self.fc(y).view(n, c, 1, 1)
        return y.expand_as(x)

class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # learnable DCT
        self.learnable_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape
        x = x * self.weight
        result = torch.sum(x, dim=[2, 3])

        return result

    def build_filter(self, pos, freq, POS):
        """
        计算DCT_WEIGHT，计算方法请参考2D-DCT变换，理论证明请参考论文FcaNet："FcaNet: Frequency Channel Attention Networks"
        :param pos:DCT滤波期间遍历当前遍历时间的滤波坐标
        :param freq:所选频率分量的坐标
        :param POS:过滤器总宽/高
        """
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):

        """
        tile_size_x：宽滤波器等于输入特征图的宽度
        tile_size_y：滤波器的高，等于高输入特性
        mapper_x：频率分量对应的X坐标
        mapper_y：频率分量对应的Y坐标
        Channel：进入特征的通道数
        """

        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)  # 创建一个与输入表征相同形状的三维片作为滤波器（2D DCT权重）

        c_part = channel // len(mapper_x)   # l (mapper_x)，每个包含一个C_PART通道
        # 一个频率分量预处理了一个特征图，一共有Len(MAPPER_X)个频率分量，所以把2D DCT filter分成Len(MAPper_x)

        # 遍历过滤器并填充过滤器的权重
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):  # 遍历过滤器的X坐标
                for t_y in range(tile_size_y):  # 遍历过滤器的Y坐标
                    # i * c_part to (i + 1) * c_part 特征是特征之一的平均值，i代表第i个
                    # #计算每个DCT_WEIGHT
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                    # T_x为方程7中滤波器(J)的X坐标，U_X为所选频率分量(W，即频率)对应的X坐标，Tile_Size_x滤波器的X轴总长(方程7中)W  )

        return nn.Parameter(dct_filter)