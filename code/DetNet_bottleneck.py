from torch import nn
import torch
class DetBolttleneck(nn.Module):
    '''extra为Ture,有1*1的卷积，否则没有,BottleneckB,否则为BottleneckA'''
    def __init__(self,in_dim,dims,stride=1,extra=False):
        super().__init__()
        self.bottleneck=nn.Sequential(  #参数是tuple,元组，也可使*list
            nn.Conv2d(in_dim,dims,1,bias=False),
            nn.BatchNorm2d(dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims,dims,3,1,2,dilation=2,bias=False),
            nn.BatchNorm2d(dims),
            nn.ReLU(dims),
            nn.Conv2d(dims,dims,1,bias=False),
            nn.BatchNorm2d(dims)
        )
        self.relu=nn.ReLU(inplace=True)
        self.extra=extra
        if extra:
            self.extra_conv=nn.Sequential(
                nn.Conv2d(in_dim,dims,1,bias=False),
                nn.BatchNorm2d(dims)
            )
    def forward(self,input):
        if self.extra:
            identity=self.extra_conv(input)
        else:
            identity=input
        out=self.bottleneck(input)
        print('{},{}'.format(identity.shape,out.shape))
        out+=identity       #按位相加时，两者的维度必须相同，特征图大小可以不等
        out=self.relu(out)
        return out

#构造一个B-A-A结构
#DetBolttleneck利用空洞卷积保证特征图的大小不变
bottleneck_b=DetBolttleneck(1024,256,1,True)
#print(bottleneck_b.extra_conv)
bottleneck_a1=DetBolttleneck(256,256)
bottleneck_a2=DetBolttleneck(256,256)
#print(bottleneck_a2.bottleneck)

input=torch.randn(1,1024,14,14)
input1=bottleneck_b(input)
print(input1.shape)
input2=bottleneck_a1(input1)     #B经过下采样之后，维度变成256
print(input2.shape)
out=bottleneck_a2(input2)
print(out.shape)