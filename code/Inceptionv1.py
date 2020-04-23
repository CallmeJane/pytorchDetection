from torch import nn
import torch.nn.functional as F  # 可以传入input
import torch


class BasicConv2D(nn.Module):
    '''定义一个包含conv与Relu的基本卷积类'''

    def __init__(self, in_dim, out_dim, kernel_size, padding=0):
        super(BasicConv2D, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, padding=padding)

    def forward(self, input):
        input = self.conv(input)
        return F.relu(input, inplace=True)


class InceptionV1(nn.Module):
    '''练习Inception搭建'''

    def __init__(self, in_dim, hid1_1, hid2_1, hid2_3, hid3_1, hid3_5, hid4_3):
        super(InceptionV1, self).__init__()
        self.batch1_1 = nn.Conv2d(in_dim, hid1_1, 1)
        self.batch3_3 = nn.Sequential(
            BasicConv2D(in_dim, hid2_1, 1),
            BasicConv2D(hid2_1, hid2_3, 3, padding=1)
        )
        self.batch5_5 = nn.Sequential(
            BasicConv2D(in_dim, hid3_1, 1),
            BasicConv2D(hid3_1, hid3_5, 5, padding=2)
        )
        self.batch3_1 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            BasicConv2D(in_dim, hid4_3, 1)
        )

    def forward(self, input):
        b1 = self.batch1_1(input)
        b2 = self.batch3_3(input)
        b3 = self.batch5_5(input)
        b4 = self.batch3_1(input)
        output = torch.cat([b1, b2, b3, b4], dim=1)  # barch_size是第一维
        return output


inceptionv1_net = InceptionV1(3, 64, 32, 64, 64, 96, 32)
#print(inceptionv1_net)
input=torch.randn(1,3,256,256)
out=inceptionv1_net(input)
print(out.shape)
