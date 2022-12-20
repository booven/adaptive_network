from networks.xception import XceptionNet
from networks.texture_net import SRnet
from components.attention import SpatialAttention, TemporalAttention, AttentionFusion, AFCA
from thop import profile
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import types


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

class RnnModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RnnModule, self).__init__()
        # self.rnn = nn.LSTM(input_size=2048, hidden_size=2048,
        #                    num_layers=1, batch_first=True)
        # self.BiLSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
        #                       num_layers=num_layers, bidirectional=True, batch_first=True)
        self.BiGRU = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bidirectional=True, batch_first=True)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.hidden_size = hidden_size

        # self.LSTM_attention = Attention(hidden_size)
        self.LSTM_attention_sum = TemporalAttention(hidden_size)

    def forward(self, x, b_z):  #[12, 1024, 10, 10]
        x = F.adaptive_avg_pool2d(self.relu(x), (1, 1)) #[12, 1024, 1, 1]
        split_fea = torch.split(x, 2, dim=0)
        out = []
        for i in range(b_z):
            temp_fea = split_fea[i]
            temp_fea = temp_fea.view(temp_fea.size(0), -1)  #[2, 1024]
            out.append(temp_fea)
        y = torch.stack(out)    #[6, 2, 1024]

        #   BiLSTM
        # outputs, hiden = self.BiLSTM(y)
        # H = outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size:]
        # r, alphas = self.LSTM_attention(H)
        # h = self.dropout(self.tanh(r))

        #   LSTM
        # r, hidden = self.rnn(y)

        #   BiGRU
        outputs, hidden = self.BiGRU(y)
        H = outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size:]
        r, alphas = self.LSTM_attention_sum(H)
        h = self.dropout(self.tanh(r))
        # print('r', r.size())

        return h

class FusionModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionModel, self).__init__()
        # self.fusion_convblk = nn.Sequential(
        #     nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2),
        #     nn.BatchNorm1d(out_channels),
        #     nn.ReLU()
        # )
        # self.Att_fusion = AFF(channels=1536)
        self.Att_fusion_2d = AttentionFusion(channels=1536)
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

class LSTM_Preprocess(nn.Module):
    def __init__(self):
        super(LSTM_Preprocess, self).__init__()

    def forward(self, x):   #   (6,2,4,256,256)
        length = x.size(0)
        split_fea = torch.split(x, 1, dim = 0)
        fea_input = split_fea[0].squeeze()
        for k in range(1, length):
            sque_squ = split_fea[k].squeeze()
            fea_input = torch.cat((fea_input, sque_squ), dim = 0)
        return fea_input

class Dimension_expansion(nn.Module):
    def __init__(self):
        super(Dimension_expansion, self).__init__()
        self.xception = XceptionNet('xception', dropout=0.5, inc=3, return_fea=True)

    def forward(self, x):
        # x = x.unsqueeze(dim=2)
        # x = self.xception.model.fea_part7(x.unsqueeze(dim=2))
        x = self.xception.model.fea_part7(x)

        return x

class Two_Stream_Net(nn.Module):
    def __init__(self, device=None ):
        super(Two_Stream_Net, self).__init__()
        self.xception_rgb = XceptionNet('xception', dropout=0.5, inc=3, return_fea=True)

        self.SR_residual = SRnet()

        self.relu = nn.ReLU(inplace=True)

        self.Channel_att_map = None
        self.Spatial_att_map = None
        # self.pa = SpatialAttention()
        self.dcta = AFCA(16,256,256)
        # self.ca = SELayer(64)

        self.sr_att = TextureAttention(16)
        self.ar_att_post = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # self.fusion = FeatureFusionModule()
        self.preprocess = LSTM_Preprocess()
        self.Xcep_Rnn = RnnModule(input_size=1024, hidden_size=1536, num_layers=1)
        self.SR_Rnn = RnnModule(input_size=1024, hidden_size=1536, num_layers=1)
        self.fusion = FusionModel(in_channels=1536, out_channels=2048)
        self.exit_flow = Dimension_expansion()

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

    def features(self, x):
        # print('x', x.size())
        #   pre process
        x = self.preprocess(x)
        # print('post',x.size())

        length = len(x)

        split_fea = torch.split(x, 3, dim=1)

        x = split_fea[0]
        y = split_fea[1]
        #
        #   textural features
        y = self.SR_residual.model.Layer_1(y)  # module.

        self.Channel_att_map = self.dcta(y)
        y = y + y * self.dcta(y)

        y = self.SR_residual.model.Layer_2_7(y)

        self.Spatial_att_map = self.sr_att(y)

        y = self.SR_residual.model.Layer_8(y)
        y = self.SR_residual.model.Layer_9(y)
        y = self.SR_residual.model.Layer_10_12(y)
        y = self.SR_residual.model.Layer_add_1(y)
        # print(y.shape)

        #   RGB features
        x = self.xception_rgb.model.fea_part1_0(x)
        x = self.xception_rgb.model.fea_part1_1(x)

        x = x * self.Spatial_att_map + x
        x = self.ar_att_post(x)

        x = self.xception_rgb.model.fea_part2(x)
        x = self.xception_rgb.model.fea_part3(x)
        x = self.xception_rgb.model.fea_part4(x)
        x = self.xception_rgb.model.fea_part5(x)

        # Rnn
        b_z = length // 2
        y = self.SR_Rnn(y, b_z)
        x = self.Xcep_Rnn(x, b_z)

        # fusion block
        fea = self.fusion(x, y)
        fea = self.exit_flow(fea)


        return fea

    def classifier(self, fea):
        out, fea = self.xception_rgb.classifier(fea)
        return out, fea

    def forward(self, x):
        '''
        x: original vector
        '''
        out, fea = self.classifier(self.features(x))

        return fea, out
        # return out, fea, self.Channel_att_map, self.Spatial_att_map

if __name__ == '__main__':
    model = Two_Stream_Net()
    dummy = torch.rand((6,2,4,299,299))
    out = model(dummy)
    flops, params = profile(model, (dummy,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))







