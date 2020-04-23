from torch import nn
import torch
class VGG(nn.Module):
    def __init__(self,num_classes=1000):
        super(VGG,self).__init__()
        layer=[]
        in_dim=3
        out_dim=64
        #循环构造卷积层，一共有13个卷积层
        for i in  range(13):
            layer+=[nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1,dilation=1)]
            in_dim=out_dim
            #再第2，4，7，10，13后增加池化层
            if i==1 or i==3 or i==6 or i==9 or i==12:
                layer+=[nn.MaxPool2d(2,2)]
                #第10个卷积层，保持和前边的通道数一致，都为512，其余加倍
                if i!=9:
                    out_dim=out_dim*2
            self.feature=nn.Sequential(*layer)
            #VGG的3个全连接层，中间有Relu与Dropout
            self.classifier=nn.Sequential(
                nn.Linear(512*7*7,4096),
                nn.ReLU(True),   #与
                nn.Dropout(),
                nn.Linear(4096,4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096,num_classes)

            )
    def forward(self,input):
        input=self.feature(input)#有在原地进行的操作
        #将特征图维度从[1,512,7,7]->[1,512*7*7]
        input=input.view(input.size(0),-1)#view(bartchsize,后几列拉成1行)
        input=self.classifier(input)
        return input

#练习使用vgg
vgg=VGG(10)
input=torch.randn(1,3,224,224)
score=vgg(input)
print(score.shape)
#可以单独使用feature
feature=vgg.feature(input)
print(feature.shape)
print(vgg.feature)
print(vgg.classifier)