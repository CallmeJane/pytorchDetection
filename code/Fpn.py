from torch import nn
import torch
import torch.nn.functional as F
#stride=2时，每一次3*3卷积都会是肯能产生小数，最后一组大小不一
class Bottleneck(nn.Module):
    '''这里选择的Backbone是ResNet'''
    expansion=4             #通道倍增
    stri=1                  #不是必须的，测试时使用
    def __init__(self,in_dim,dims,stride=1,downsample=None):
        super(Bottleneck,self).__init__()
        self.stri=stride
        self.bottleblock=nn.Sequential(
            nn.Conv2d(in_dim,dims,1,bias=False),
            nn.BatchNorm2d(dims),
            nn.ReLU(True),
            nn.Conv2d(dims,dims,3,stride,padding=1,bias=False),     #特征图变小
            nn.BatchNorm2d(dims),
            nn.ReLU(True),
            nn.Conv2d(dims,dims*self.expansion,1,bias=False),
            nn.BatchNorm2d(dims*self.expansion)
        )
        self.relu=nn.ReLU(True)
        self.down_sample=downsample
    def forward(self,input):
        idesity=input
        out=self.bottleblock(input)
        if self.down_sample is not None:
            idesity=self.down_sample(input)  #当stride=2时，两者虽然维度相同，但是特征图大小不同
        #a=torch.Tensor([[1,2],[3,4]])
        #b=torch.Tensor([[1],[3]])
        #print(a+b)tensor([[2., 3.],[6., 7.]])
        if self.stri==2:
            print('bottleblock out={},idesity={}'.format(out.shape,idesity.shape))#bottleblock out={} torch.Size([1, 256, 56, 56])
        out+=idesity        #按照元素相加
        out = self.relu(out)
        return out

class FPN(nn.Module):
    '''
    练习FPN模块
    layers:list,每一阶段Bottleneck的数量
    '''
    def __init__(self,layers):      #认为一般架构是这样的
        super().__init__()
        self.indim=64
        #处理c1的模块
        self.conv1=nn.Conv2d(3,64,7,2,3,bias=False)
        self.bn=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(True)
        self.maxpool=nn.MaxPool2d(3,2,1)
        #搭建自上而下的c2,c3,c4,c5
        self.layer1=self._make_layer(64,layers[0]) #注意是stride=2,用于特征图的变小,输出维度是64*4=256
        self.layer2=self._make_layer(128,layers[1],2) #第一个参数用于控制输出维度，输出维度128*4=512
        self.layer3=self._make_layer(256,layers[2],2)  #256*4=
        self.layer4=self._make_layer(512,layers[3],2)  #512*4=2048
        #c5减少通道数变成p5
        self.toplayer=nn.Conv2d(2048,256,1,1,0)
        #3*3卷积融合
        self.smooth1=nn.Conv2d(256,256,3,1,1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)
        #横向连接
        self.latlayer1=nn.Conv2d(1024,256,1,1,0)
        self.latlayer2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(256, 256, 1, 1, 0)
    #自上而下的上采样模块
    def _upsample_add(self,input,input2):
        '''
        :param input: 自下而上的输入，p*，是经过多层卷积后得到的
        :param input2: 横向输入,且横向输入的特征比纵向输入的大
        :return: 两者的上采样结果
        '''
        _,_,H,W=input2.shape
        print('input{},output{}'.format(input.shape,input2.shape))
        return F.interpolate(input,size=(H,W),mode='bilinear')+input2
        #return F.upsample(input,size=(H,W),mode='bilinear')+input2
    def forward(self,input):
        c1=self.maxpool(self.relu(self.bn(self.conv1(input))))
        c2=self.layer1(c1)
        c3=self.layer2(c2)
        c4=self.layer3(c3)
        c5=self.layer4(c4)
        p5=self.toplayer(c5)
        p4=self._upsample_add(p5,self.latlayer1(c4))
        p3=self._upsample_add(p4,self.latlayer2(c3))
        p2=self._upsample_add(p3,self.latlayer3(c2))
        #卷积融合平滑处理
        p4=self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2,p3,p4,p5

    def _make_layer(self,dims,blocks,stride=1):
        '''
        构造c2,c3,c4,c5
        dims:每层Bottleneck的dims,和Bottleneck.expansion共同控制输出维度
        blcok:每个层包含的Bottleneck的个数
        stride:1表示不改变特征图大小，2表示特征图缩小
        '''
        downsample=None
        if stride!=1 or self.indim!=Bottleneck.expansion*dims:
            #输入输出通道数不同，要增加下采样层
            downsample=nn.Sequential(
                nn.Conv2d(self.indim,Bottleneck.expansion*dims,1,stride=stride,bias=False),#都是对self.indim进行
                nn.BatchNorm2d(Bottleneck.expansion*dims)
            )
        layers=[]
        layers.append(Bottleneck(self.indim,dims,stride,downsample))
        self.indim=dims*Bottleneck.expansion      #更新indim，后几个block的输出都是,dims*Bottleneck.expansion
        for i in range(1,blocks):
            #后几个模块不需要下采样
            layers.append(Bottleneck(self.indim,dims))
        return nn.Sequential(*layers)    #不能直接返回lsit，而是要返回一个模块

netFPB=FPN([3,4,6,3])
#print(netFPB.layer1)
input=torch.randn(1,3,224,224)
out=netFPB(input)
print('p2={},p3={},p4={},p5={}'.format(out[0].shape,out[1].shape,out[2].shape,out[3].shape))