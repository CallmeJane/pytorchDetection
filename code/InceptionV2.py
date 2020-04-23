import torch
from torch import nn
import torch.nn.functional as F


class BasicConv2Dv2(nn.Module):
    '''构建基础卷积模块，与v1相比，增加了BN'''

    def __init__(self, in_dim, out_dim, kernel_size, padding=0):
        super(BasicConv2Dv2, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_dim, eps=0.01)

    def forward(self, input):
        input = self.conv(input)
        input = self.bn(input)
        return F.relu(input, inplace=True)


class InceptionV2(nn.Module):
    '''将5*5拆分成两个3*3'''

    def __init__(self, in_dim, hid1_1, hid2_1, hid2_3, hid3_1, hid3_3_1, hid3_3_2, hid4_3):
        super(InceptionV2, self).__init__()
        self.batch1_1 = nn.Conv2d(in_dim, hid1_1, 1)
        self.batch3_3 = nn.Sequential(
            BasicConv2Dv2(in_dim, hid2_1, 1),
            BasicConv2Dv2(hid2_1, hid2_3, 3,padding=1)
        )
        self.batch5_5 = nn.Sequential(
            BasicConv2Dv2(in_dim, hid3_1, 1),
            BasicConv2Dv2(hid3_1, hid3_3_1, 3, padding=1),
            BasicConv2Dv2(hid3_3_1, hid3_3_2, 3, padding=1)
        )
        self.batchPooling = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2Dv2(in_dim, hid4_3, 1)
        )

    def forward(self, input):
        b1 = self.batch1_1(input)
        b2 = self.batch3_3(input)
        b3 = self.batch5_5(input)
        b4 = self.batchPooling(input)
        output = torch.cat([b1, b2, b3, b4], dim=1)
        return output


inceptionv2_net = InceptionV2(192, 96, 48, 64, 64, 96, 96, 64)
input=torch.randn(1,192,32,32)
out=inceptionv2_net(input)
print(out.shape)#torch.Size([1, 320, 32, 32])
