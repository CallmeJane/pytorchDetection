import torch
import torch.nn as nn
import torch.nn.functional as F

def ConvLayer():
    '''卷积层练习'''
    #搭建卷积层,主要是卷积核的设置
    conv=nn.Conv2d(in_channels=1,out_channels=2,kernel_size=3,stride=1,padding=1,dilation=1,
              groups=1,bias=True)
    print(conv.weight.shape)   #2*1*5*5
    print(conv.bias)
    print(conv.bias.shape)     #[2],1*2,1行2列
    #第一维表示训练的batch
    input=torch.ones(1,1,5,5)  #1*1*5*5
    output=conv(input)
    print(output.shape)   #1*2*5*5
    print(output)

def AFSigmoid():
    '''练习simgoid的使用,但对应的梯度函数两侧为0'''
    input=torch.ones(1,1,2,2)
    #sigmoid实例化
    sigmoid=nn.Sigmoid()
    #输入参数
    out=sigmoid(input)
    print(out)

def AFReLu():
    '''练习ReLu函数的使用'''
    input=torch.randn(1,1,2,2)
    print(input)
    #Relu实例化,inplace是直接在输入上进行计算，将结果还是放大输入上，来节省内存
    relu=nn.ReLU(inplace=True)
    relu(input)
    print(input)
def AFleakyRelu():
    '''是对Relu的改进，但是实际上没有预想的效果那么好'''
    input=torch.randn(1,1,2,2)
    print(input)
    lrelu=nn.LeakyReLU()
    out=lrelu(input)
    print(out)
def AFSoftmax():
    '''练习softmax的使用'''
    input=torch.randn(1,4)
    print(input)
    out=F.softmax(input,1)
    print(out)

def EXPooling():
    '''练习Pooling的使用'''
    input=torch.randn(1,1,4,4)
    print(input)
    maxPooling=nn.MaxPool2d(2,stride=2)
    avgPooling=nn.AvgPool2d(2,stride=2)
    out1=maxPooling(input)
    print(out1)
    out2=avgPooling(input)
    print(out2)
def EXDropOut():
    '''练习dropout,out中一部分设置为0'''
    dropout=nn.Dropout(0.5,inplace=False)
    input=torch.randn(1,10)
    out=dropout(input)   #tensor([[-0.0000,  2.1541,  0.7889,  0.1085, -0.5150,  0.4455,  1.0463,  0.0000,
              #1.6154, -0.0000]])
    print(out)
def EXBatchNormal():
    '''练习batchNormal的使用'''
    #num_features,特征的通道数
    bn=nn.BatchNorm2d(num_features=64)
    print(bn)#BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    input=torch.randn(4,64,224,224)
    out=bn(input)#不改变原来的大小
    print(out.shape)

def EXFullyConnect():
    '''全连接层的练习'''
    input=torch.randn(4,1024)
    linear=nn.Linear(1024,4096)
    out=linear(input)
    print(out.shape)
def EXDilation():
    '''练习空洞卷积'''
    conv1=nn.Conv2d(in_channels=3,out_channels=256,kernel_size=3,stride=1,padding=0,dilation=1)
    print(conv1)#Conv2d(3, 256, kernel_size=(3, 3), stride=(1, 1))
    conv2=nn.Conv2d(in_channels=3,out_channels=256,kernel_size=3,stride=1,padding=0,dilation=2)
    print(conv2)#Conv2d(3, 256, kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2))

print(nn.MaxPool2d(2,2))
conv=torch.ones(1,512,7,7)
print(conv.size(0))
print(conv.view(conv.size(0),-1))
print(conv[0].shape)
