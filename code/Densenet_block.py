import torch
from torch import nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    '''练习Desnet的残差模块，Bottleneck'''

    def __init__(self, in_dim, growth_rate):
        super(Bottleneck, self).__init__()
        inter_dim = growth_rate * 4
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim, inter_dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_dim)
        self.conv2 = nn.Conv2d(inter_dim, growth_rate, 3, padding=1, bias=False)

    def forward(self, input):
        #先激活函数，后卷积
        out = self.conv1(F.relu(self.bn1(input)))
        out = self.conv2(F.relu(self.bn2(out)))
        out=torch.cat((input,out),dim=1)
        return out
class Denseblock(nn.Module):
    '''练习DesNet的基础模块，DenseBlock'''
    def __init__(self,in_dim,growth_rate,n_denseblocks):
        super(Denseblock,self).__init__()
        layer=[]
        for i in range(n_denseblocks):
            layer.append(Bottleneck(in_dim,growth_rate))
            in_dim+=growth_rate
        self.denseblock=nn.Sequential(*layer)
    def forward(self,input):
        out = self.denseblock(input)
        return out
class Transition(nn.Module):
    '''练习DenseBlock之后的Transition'''
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.bn=nn.BatchNorm2d(in_dim)
        self.conv=nn.Conv2d(in_dim,out_dim,1)  #降维
        self.avg_pooling=nn.AvgPool2d(2)
    def forward(self,input):
        out=self.bn(input)
        out=F.relu(out)
        out=self.conv(out)
        out=self.avg_pooling(out)
        return out

denseblock=Denseblock(64,32,6)
input=torch.randn(1,64,32,32)
out=denseblock(input)
print(out.shape)#6的Desblock模块，torch.Size([1, 256, 32, 32])
