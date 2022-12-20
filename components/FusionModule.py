import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import torch.fft as afft
import types

class FusionModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionModel, self).__init__()
        # self.fusion_convblk = nn.Sequential(
        #     nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2),
        #     nn.BatchNorm1d(out_channels),
        #     nn.ReLU()
        # )
        # self.Att_fusion = AFF(channels=1536)
        self.Att_fusion_2d = AFF_2d(channels=1536)
        # self.BilinearPooling = CompactBilinearPooling(2048, 2048, 2048)
        self.init_weight()

    def forward(self, x, y):
        #   concat
        # fuse_fea = torch.cat((x, y), dim=2)
        # fuse_fea = self.fusion_convblk(fuse_fea.permute(0, 2, 1))
        # fuse_fea = fuse_fea.squeeze()

        #   add
        # fuse_fea = x.add(y)

        #  Attentional Feature Fusion
        x = x.unsqueeze(dim=2)
        y = y.unsqueeze(dim=2)

        # fuse_fea = self.Att_fusion(x.permute(0, 2, 1), y.permute(0, 2, 1))
        fuse_fea = self.Att_fusion_2d(x.unsqueeze(dim=2), y.unsqueeze(dim=2))
        # fuse_fea = fuse_fea.sum(dim=2)

        #   Bilinear Pooling
        # x = x.sum(dim=1)
        # y = y.sum(dim=1)
        # x = x.unsqueeze(dim=2)
        # y = y.unsqueeze(dim=2)
        # fuse_fea = self.BilinearPooling(x.unsqueeze(dim=2), y.unsqueeze(dim=2))

        return fuse_fea

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv1d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


#   双线性池化，参考自：https://github.com/jnhwkim/cbp
#   相关paper："Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding"

class CompactBilinearPooling(nn.Module):
    """
    Compute compact bilinear pooling over two bottom inputs.
    Args:
        output_dim: output dimension for compact bilinear pooling.
        sum_pool: (Optional) If True, sum the output along height and width
                  dimensions and return output shape [batch_size, output_dim].
                  Otherwise return [batch_size, height, width, output_dim].
                  Default: True.
        rand_h_1: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_1`
                  if is None.
        rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_1`. Automatically generated from `seed_s_1` if is
                  None.
        rand_h_2: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_2`
                  if is None.
        rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_2`. Automatically generated from `seed_s_2` if is
                  None.
    """

    def __init__(self, input_dim1, input_dim2, output_dim,
                 sum_pool=True, rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None):
        super(CompactBilinearPooling, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sum_pool = sum_pool

        if rand_h_1 is None:
            np.random.seed(1)
            rand_h_1 = np.random.randint(output_dim, size=self.input_dim1)
        if rand_s_1 is None:
            np.random.seed(3)
            rand_s_1 = 2 * np.random.randint(2, size=self.input_dim1) - 1

        self.sparse_sketch_matrix1 = Variable(self.generate_sketch_matrix(
            rand_h_1, rand_s_1, self.output_dim))

        if rand_h_2 is None:
            np.random.seed(5)
            rand_h_2 = np.random.randint(output_dim, size=self.input_dim2)
        if rand_s_2 is None:
            np.random.seed(7)
            rand_s_2 = 2 * np.random.randint(2, size=self.input_dim2) - 1

        self.sparse_sketch_matrix2 = Variable(self.generate_sketch_matrix(
            rand_h_2, rand_s_2, self.output_dim))

    def forward(self, bottom1, bottom2):
        """
        bottom1: 1st input, 4D Tensor of shape [batch_size, input_dim1, height, width].
        bottom2: 2nd input, 4D Tensor of shape [batch_size, input_dim2, height, width].
        """
        assert bottom1.size(1) == self.input_dim1 and \
            bottom2.size(1) == self.input_dim2

        batch_size, _, height, width = bottom1.size()

        bottom1_flat = bottom1.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim1)
        bottom2_flat = bottom2.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim2)

        sketch_1 = bottom1_flat.mm(self.sparse_sketch_matrix1.to(device))
        sketch_2 = bottom2_flat.mm(self.sparse_sketch_matrix2.to(device))

        fft1 = afft.fft(sketch_1)
        fft2 = afft.fft(sketch_2)

        fft_product = fft1 * fft2

        cbp_flat = afft.ifft(fft_product).real

        cbp = cbp_flat.view(batch_size, height, width, self.output_dim)

        if self.sum_pool:
            cbp = cbp.sum(dim=1).sum(dim=1)

        return cbp

    @staticmethod
    def generate_sketch_matrix(rand_h, rand_s, output_dim):
        """
        Return a sparse matrix used for tensor sketch operation in compact bilinear
        pooling
        Args:
            rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
            rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
            output_dim: the output dimensions of compact bilinear pooling.
        Returns:
            a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
        """

        # Generate a sparse matrix for tensor count sketch
        rand_h = rand_h.astype(np.int64)
        rand_s = rand_s.astype(np.float32)
        assert(rand_h.ndim == 1 and rand_s.ndim ==
               1 and len(rand_h) == len(rand_s))
        assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))

        input_dim = len(rand_h)
        indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                                  rand_h[..., np.newaxis]), axis=1)
        indices = torch.from_numpy(indices)
        rand_s = torch.from_numpy(rand_s)
        sparse_sketch_matrix = torch.sparse.FloatTensor(
            indices.t(), rand_s, torch.Size([input_dim, output_dim]))
        return sparse_sketch_matrix.to_dense()


#   注意力特征融合，参考自：https://github.com/YimianDai/open-aff
#   相关paper："Attentional Feature Fusion"

class AFF_2d(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels, r=4):
        super(AFF_2d, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)

        return xo