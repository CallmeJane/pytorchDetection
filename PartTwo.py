import torch
#主要用到
# torch.nn,torch.optim
# torch.util.data,
# torchvision.model,torchvision.transformer
#tensorboardX
from torch import nn
from torch import optim
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter   #用于可视化
#http://www.bubuko.com/infodetail-3460794.html(visdom启动server)
#https://blog.csdn.net/qq_43280818/article/details/104241744（下载static缺失文件）
import visdom

#使用原始方法创建网络,要求：继承mm.Model和实现forward函数
class Linear(nn.Module):        #只包含一层，输入和输出
    def __init__(self,in_dim,out_dim):
        super(Linear,self).__init__()
        #需要学习的参数,y=wx+b,根据维度计算的
        self.w=nn.Parameter(torch.randn(in_dim,out_dim))
        self.b=nn.Parameter(torch.randn(out_dim))
    def forward(self, x):
        x=x.matmul(self.w)
        y=x+self.b
        return y
#两层网络
class RawPerception(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim):
        super(RawPerception,self).__init__()
        self.layer1=Linear(in_dim,hid_dim)
        self.layer2=Linear(hid_dim,out_dim)
    def forward(self,x):
        x=self.layer1(x)
        y=torch.sigmoid(x)
        y=self.layer2(y)
        y=torch.sigmoid(y)
        return y
#利用nn.Sequential快速创建网络，Perception
class Perception(nn.Module):
    def  __init__(self,in_dim,hid_dim,out_dim):
        super(Perception,self).__init__()
        #利用sequence来快速搭建网络
        self.layer=nn.Sequential(
            nn.Linear(in_dim,hid_dim),
            nn.Sigmoid(),
            nn.Linear(hid_dim,out_dim),
            nn.Sigmoid())
    def forward(self,x):
        y=self.layer(x)
        return y

#利用nn.Sequential快速创建网络
class MLP(nn.Module):
    def __init__(self,in_dim,hid_dim1,hid_dim2,out_dim):
        super(MLP,self).__init__()
        #快速搭3层感知机
        self.layer=nn.Sequential(
            nn.Linear(in_dim,hid_dim1),
            nn.ReLU(),
            nn.Linear(hid_dim1,hid_dim2),
            nn.ReLU(),
            nn.Linear(hid_dim2,out_dim),
            nn.ReLU()
        )
    def forward(self,x):
        y = self.layer(x)
        return y

#数据加载
class MyData(Dataset):
    def __init__(self,image_path,anno_path,transformer=None):
        super(MyData,self).__init__()
        #初始化，读取数据
    def __len__(self):
        '''获得数据集的总大小'''
        pass
    def __getitem__(self, item):
        '''对于指定的iterm,获取，并且返回'''
        pass
#datas=MyData("image_path","anno_path",transformer=transforms.Compose(
#    [transforms.Resize(256),transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]
#))      #实例化
##可迭代对象，分批训练
#dataloader=DataLoader(datas,batch_size=4,shuffle=True,num_workers=4)
#dataIter=iter(dataloader)
#iters_per_epoch=10
#for step in range(iters_per_epoch):
#    data=next(dataIter)
#    #data用于训练数据
#
##创建writer对象,数据可视化
#writer=SummaryWriter('logs/tmp')

#创建visdom客户端,env对可视化空间分区
vis=visdom.Visdom(env='first')
#写一句text文本，win是代表的显示的窗格
vis.text('first visdom',win='text1')
#在上一窗格追加一句,append=True,表示之前的不会被覆盖
vis.text('append first visdom',win='text1',append=True)
#opt可以进行标题，坐标轴，标签的配置
for i in range(20):
    vis.line(X=torch.FloatTensor([i]),Y=torch.FloatTensor([-i**2+20*i+1]),
                                 opts={'title':'y=-i^2+20*i+1'},win='loss',
                                 update='append')
#随机生成一张图片
vis.image(torch.randn(3,256,256),win='random_image')

