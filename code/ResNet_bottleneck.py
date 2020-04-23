from torch import  nn
import torch
class Bottleneck(nn.Module):
    '''练习残差网络,利用下采样的方式，不止这一种'''
    def __init__(self,in_dim,out_dim,stride=1):
        super(Bottleneck,self).__init__()
        self.bottleneck=nn.Sequential(
            nn.Conv2d(in_dim,in_dim,1,bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim,in_dim,3,padding=1,bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim,out_dim,1,bias=False),
            nn.BatchNorm2d(out_dim)
        )
        self.relu=nn.ReLU(inplace=True)
        #下采样包含一个1*1卷积和BN
        self.downSampel=nn.Sequential(
            nn.Conv2d(in_dim,out_dim,1,1),
            nn.BatchNorm2d(out_dim)
        )
    def forward(self,input):
        #先做一个恒等变形
        idensity=input
        out = self.bottleneck(input)
        print('{},{},{}'.format(id(input), id(out),id(idensity)))
        idensity=self.downSampel(input)
        out+=idensity
        out=self.relu(out)
        return out

#测试ReLU(inplace=False)的参数
relu=nn.ReLU(inplace=False)
input=torch.randn(2,2)
print(input)
out=relu(input)          #当inplace=False，out和input不同，否则两者相同

bottleneck1=Bottleneck(64,256)
input=torch.randn(1,64,28,28)
out=bottleneck1(input)
#相比输入，输出的分辨路没有变，通道数变成4倍
print(out.shape)#torch.Size([1, 256, 28, 28])
