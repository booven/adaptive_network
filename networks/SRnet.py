"""
Code from SRnet:https://ieeexplore.ieee.org/document/8470101
code：http://dde.binghamton.edu/download/feature_extractors/
由于本网络对骨干网络的特殊要求，故在layer调用部分进行了调整。请注意，为了与其他方法更好的比对，我们并没有改动任何layer里面的内容。
"""
import os
import argparse
import torch.backends.cudnn as cudnn
from thop import profile
import torch
# import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
# from lib.nets.xception import xception
import math
import torchvision
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
# gpu_ids = [*range(osenvs)]
PRETAINED_WEIGHT_PATH = '../SRNet_model_weights.pt'


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class SR_Net(nn.Module):
    """

    """

    def __init__(self, inc = 1):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(SR_Net, self).__init__()
        # Layer 1
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=64,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Layer 2
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=16,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        # Layer 3
        self.layer31 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(16)
        self.layer32 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(16)
        # Layer 4
        self.layer41 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(16)
        self.layer42 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn42 = nn.BatchNorm2d(16)
        # Layer 5
        self.layer51 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn51 = nn.BatchNorm2d(16)
        self.layer52 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn52 = nn.BatchNorm2d(16)
        # Layer 6
        self.layer61 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn61 = nn.BatchNorm2d(16)
        self.layer62 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn62 = nn.BatchNorm2d(16)
        # Layer 7
        self.layer71 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn71 = nn.BatchNorm2d(16)
        self.layer72 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn72 = nn.BatchNorm2d(16)
        # Layer 8
        self.layer81 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=1, stride=2, padding=0, bias=False)
        self.bn81 = nn.BatchNorm2d(16)
        self.layer82 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn82 = nn.BatchNorm2d(16)
        self.layer83 = nn.Conv2d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn83 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Layer 9
        self.layer91 = nn.Conv2d(in_channels=16, out_channels=64,
                                 kernel_size=1, stride=2, padding=0, bias=False)
        self.bn91 = nn.BatchNorm2d(64)
        self.layer92 = nn.Conv2d(in_channels=16, out_channels=64,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn92 = nn.BatchNorm2d(64)
        self.layer93 = nn.Conv2d(in_channels=64, out_channels=64,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn93 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Layer 10
        self.layer101 = nn.Conv2d(in_channels=64, out_channels=128,
                                  kernel_size=1, stride=2, padding=0, bias=False)
        self.bn101 = nn.BatchNorm2d(128)
        self.layer102 = nn.Conv2d(in_channels=64, out_channels=128,
                                  kernel_size=3, stride=1, padding=1, bias=False)
        self.bn102 = nn.BatchNorm2d(128)
        self.layer103 = nn.Conv2d(in_channels=128, out_channels=128,
                                  kernel_size=3, stride=1, padding=1, bias=False)
        self.bn103 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Layer 11
        self.layer111 = nn.Conv2d(in_channels=128, out_channels=256,
                                  kernel_size=1, stride=2, padding=0, bias=False)
        self.bn111 = nn.BatchNorm2d(256)
        self.layer112 = nn.Conv2d(in_channels=128, out_channels=256,
                                  kernel_size=3, stride=1, padding=1, bias=False)
        self.bn112 = nn.BatchNorm2d(256)
        self.layer113 = nn.Conv2d(in_channels=256, out_channels=256,
                                  kernel_size=3, stride=1, padding=1, bias=False)
        self.bn113 = nn.BatchNorm2d(256)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 12
        self.layer121 = nn.Conv2d(in_channels=256, out_channels=512,
                                  kernel_size=3, stride=2, padding=1, bias=False)
        self.bn121 = nn.BatchNorm2d(512)

        self.layer122 = SeparableConv2d(512, 1024, 3, 1, 1)
        self.bn122 = nn.BatchNorm2d(1024)
        self.layer123 = SeparableConv2d(1024, 2048, 3, 1, 1)
        self.bn123 = nn.BatchNorm2d(2048)

        # Fully Connected layer
        self.fc = nn.Linear(2048 * 1 * 1, 1)



    def Layer_1(self, inputs):
        # Layer 1
        conv = self.layer1(inputs)
        actv = F.relu(self.bn1(conv))
        # print(actv.shape)
        # Layer 2
        conv = self.layer2(actv)
        actv = F.relu(self.bn2(conv))
        # print(actv.shape)

        return actv

    def Layer_2_7(self, actv):

        # Layer 3
        conv1 = self.layer31(actv)
        actv1 = F.relu(self.bn31(conv1))
        conv2 = self.layer32(actv1)
        bn = self.bn32(conv2)
        res = torch.add(actv, bn)
        # Layer 4
        conv1 = self.layer41(res)
        actv1 = F.relu(self.bn41(conv1))
        conv2 = self.layer42(actv1)
        bn = self.bn42(conv2)
        res = torch.add(res, bn)
        # Layer 5
        conv1 = self.layer51(res)
        actv1 = F.relu(self.bn51(conv1))
        conv2 = self.layer52(actv1)
        bn = self.bn52(conv2)
        res = torch.add(res, bn)
        # Layer 6
        conv1 = self.layer61(res)
        actv1 = F.relu(self.bn61(conv1))
        conv2 = self.layer62(actv1)
        bn = self.bn62(conv2)
        res = torch.add(res, bn)
        # Layer 7
        conv1 = self.layer71(res)
        actv1 = F.relu(self.bn71(conv1))
        conv2 = self.layer72(actv1)
        bn = self.bn72(conv2)
        res = torch.add(res, bn)

        return res

    def Layer_8(self, res):
        # Layer 8
        convs = self.layer81(res)
        convs = self.bn81(convs)
        conv1 = self.layer82(res)
        actv1 = F.relu(self.bn82(conv1))
        conv2 = self.layer83(actv1)
        bn = self.bn83(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)

        return res

    def Layer_9(self, res):
        convs = self.layer91(res)
        convs = self.bn91(convs)
        conv1 = self.layer92(res)
        actv1 = F.relu(self.bn92(conv1))
        conv2 = self.layer93(actv1)
        bn = self.bn93(conv2)
        pool = self.pool2(bn)
        res = torch.add(convs, pool)

        return res

    def Layer_10_12(self, res):

        # Layer 10
        convs = self.layer101(res)
        convs = self.bn101(convs)
        conv1 = self.layer102(res)
        actv1 = F.relu(self.bn102(conv1))
        conv2 = self.layer103(actv1)
        bn = self.bn103(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)
        # Layer 11
        convs = self.layer111(res)
        convs = self.bn111(convs)
        conv1 = self.layer112(res)
        actv1 = F.relu(self.bn112(conv1))
        conv2 = self.layer113(actv1)
        bn = self.bn113(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)
        # print(res.shape)
        # # Layer 12
        # print(res.shape)
        conv1 = self.layer121(res)
        bn = self.bn121(conv1)

        return bn

    def Layer_add_1(self, bn):
        actv1 = F.relu(bn)
        conv1 = self.layer122(actv1)
        bn = self.bn122(conv1)
        return bn

    def Layer_add_2(self, bn):

        actv1 = F.relu(bn)
        conv2 = self.layer123(actv1)
        bn = self.bn123(conv2)
        # print(bn.shape)
        return bn

    def Layer_last(self, bn):
        avgp = torch.mean(bn, dim=(2, 3), keepdim=True)
        # fully connected
        flatten = avgp.view(avgp.size(0), -1)
        # print("flatten:", flatten.shape)
        fc = self.fc(flatten)
        # print("FC:",fc.shape)
        out = F.log_softmax(fc, dim=1)

        return fc, flatten


    def forward(self, input):
        out = self.Layer_1(input)
        out = self.Layer_2_7(out)
        out = self.Layer_8(out)
        out = self.Layer_9(out)
        out = self.Layer_10_12(out)
        # flatten, out = self.Layer_last(out)

        return out


def sr_net(num_classes=1000,inc=1):
    model = SR_Net(inc=1)

    return model


class SR_Net(nn.Module):
    """
    """

    def __init__(self):
        super(Pre_Train_SR, self).__init__()

        def return_pytorch04_xception(pretrained=True):
            # Raises warning "src not broadcastable to dst" but thats fine
            model = sr_net()
            # if pretrained:
            #     model = nn.DataParallel(model, device_ids= gpu_ids)
            #     # print(model)
            #     cudnn.benchmark = True
            #     state_dict = torch.load(PRETAINED_WEIGHT_PATH)
            #     # print(state_dict)
            #     # for name, weights in state_dict.items():
            #     #     if 'pointwise' in name:
            #     #         state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
            #     model.load_state_dict(state_dict, False)
            #     # model.load_state_dict({k.replace('module.', ''): v for k, v in
            #     #                        torch.load(PRETAINED_WEIGHT_PATH).items()})
            return model
        self.model = return_pytorch04_xception()

    def forward(self, x):
        out = self.model(x)
        return out

    def Layer_last(self, x):
        out, x = self.model.Layer_last(x)
        return out, x

if __name__ == '__main__':
    model = Pre_Train_SR( )
    # print(model)
    # model = model.cuda()
    # from torchsummary import summary
    # input_s = (3, image_size, image_size)
    # print(summary(model, input_s))
    dummy = torch.rand(10, 1, 256, 256)
    out = model(dummy)

    # flops, params = profile(model, (dummy,))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    # print(out)
    # print(out.size())
    # x = model.features(dummy)
    # out, x = model.classifier(x)
    # print(out.size())
    # print(x.size())
