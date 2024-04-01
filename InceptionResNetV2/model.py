import torch
import torch.nn as nn
import torch.nn.functional as F

def Conv1(in_channel,out_channel,kernel_size,stride,padding):
    return nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding),
                         nn.BatchNorm2d(out_channel),
                         nn.ReLU())

class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.conv1=Conv1(in_channel=3,out_channel=32,kernel_size=3,stride=2,padding=0)
        self.conv2=Conv1(in_channel=32,out_channel=32,kernel_size=3,stride=1,padding=0)
        self.conv3=Conv1(in_channel=32,out_channel=64,kernel_size=3,stride=1,padding=1)
        self.branch1_1=nn.MaxPool2d(kernel_size=3,stride=2)
        self.branch1_2=Conv1(in_channel=64,out_channel=96,kernel_size=3,stride=2,padding=0)
        self.branch2_1_1=Conv1(in_channel=160,out_channel=64,kernel_size=1,stride=1,padding=0)
        self.branch2_1_2=Conv1(in_channel=64,out_channel=96,kernel_size=3,stride=1,padding=0)
        self.branch2_2_1=Conv1(in_channel=160,out_channel=64,kernel_size=1,stride=1,padding=0)
        self.branch2_2_2=Conv1(in_channel=64,out_channel=64,kernel_size=(7,1),stride=1,padding=(3,0))
        self.branch2_2_3=Conv1(in_channel=64,out_channel=64,kernel_size=(1,7),stride=1,padding=(0,3))
        self.branch2_2_4=Conv1(in_channel=64,out_channel=96,kernel_size=3,stride=1,padding=0)
        self.branch3_1=Conv1(in_channel=192,out_channel=192,kernel_size=3,stride=2,padding=0)
        self.branch3_2=nn.MaxPool2d(kernel_size=3,stride=2)

    def forward(self,x):
        out1=self.conv1(x)
        out2=self.conv2(out1)
        out3=self.conv3(out2)
        out4_1=self.branch1_1(out3)
        out4_2=self.branch1_2(out3)
        out4=torch.cat((out4_1,out4_2),dim=1)
        out5_1=self.branch2_1_2(self.branch2_1_1(out4))
        out5_2=self.branch2_2_4(self.branch2_2_3(self.branch2_2_2(self.branch2_2_1(out4))))
        out5=torch.cat((out5_1,out5_2),dim=1)
        out6_1=self.branch3_1(out5)
        out6_2=self.branch3_2(out5)
        out=torch.cat((out6_1,out6_2),dim=1)
        return out
    
class InceptionResNetA(nn.Module):
    def __init__(self,in_channel,scale=0.1):
        super(InceptionResNetA, self).__init__()
        self.branch1=Conv1(in_channel=in_channel,out_channel=32,kernel_size=1,stride=1,padding=0)
        self.branch2_1=Conv1(in_channel=in_channel,out_channel=32,kernel_size=1,stride=1,padding=0)
        self.branch2_2=Conv1(in_channel=32,out_channel=32,kernel_size=3,stride=1,padding=1)
        self.branch3_1=Conv1(in_channel=in_channel,out_channel=32,kernel_size=1,stride=1,padding=0)
        self.branch3_2=Conv1(in_channel=32,out_channel=48,kernel_size=3,stride=1,padding=1)
        self.branch3_3=Conv1(in_channel=48,out_channel=64,kernel_size=3,stride=1,padding=1)
        self.linear=Conv1(in_channel=128,out_channel=384,kernel_size=1,stride=1,padding=0)
        self.out_channel=384
        self.scale=scale

        self.shortcut=nn.Sequential()
        if in_channel != self.out_channel:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels=in_channel,out_channels=self.out_channel,kernel_size=1,stride=1,padding=0),
            )

    def forward(self,x):
        output1=self.branch1(x)
        output2=self.branch2_2(self.branch2_1(x))
        output3=self.branch3_3(self.branch3_2(self.branch3_1(x)))
        out=torch.cat((output1,output2,output3),dim=1)
        out=self.linear(out)
        x=self.shortcut(x)
        out=x+self.scale*out
        out=F.relu(out)
        return out
    
class InceptionResNetB(nn.Module):
    def __init__(self,in_channel,scale=0.1):
        super(InceptionResNetB, self).__init__()
        self.branch1=Conv1(in_channel=in_channel,out_channel=192,kernel_size=1,stride=1,padding=0)
        self.branch2_1=Conv1(in_channel=in_channel,out_channel=128,kernel_size=1,stride=1,padding=0)
        self.branch2_2=Conv1(in_channel=128,out_channel=160,kernel_size=(1,7),stride=1,padding=(0,3))
        self.branch2_3=Conv1(in_channel=160,out_channel=192,kernel_size=(7,1),stride=1,padding=(3,0))
        self.linear=Conv1(in_channel=384,out_channel=1152,kernel_size=1,stride=1,padding=0)
        self.out_channel=1152
        self.scale=scale

        self.shortcut=nn.Sequential()
        if in_channel != self.out_channel:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels=in_channel,out_channels=self.out_channel,kernel_size=1,stride=1,padding=0)
            )

    def forward(self,x):
        output1=self.branch1(x)
        output2=self.branch2_3(self.branch2_2(self.branch2_1(x)))
        out=torch.cat((output1,output2),dim=1)
        out=self.linear(out)
        x=self.shortcut(x)
        out=x+out*self.scale
        out=F.relu(out)
        return out

class InceptionResNetC(nn.Module):
    def __init__(self,in_channel,scale=0.1):
        super(InceptionResNetC, self).__init__()
        self.branch1 = Conv1(in_channel=in_channel, out_channel=192, kernel_size=1, stride=1, padding=0)
        self.branch2_1 = Conv1(in_channel=in_channel, out_channel=192, kernel_size=1, stride=1, padding=0)
        self.branch2_2 = Conv1(in_channel=192, out_channel=224, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3 = Conv1(in_channel=224, out_channel=256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.linear = Conv1(in_channel=448, out_channel=2144, kernel_size=1, stride=1, padding=0)
        self.out_channel = 2144
        self.scale = scale

        self.shortcut = nn.Sequential()
        if in_channel != self.out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x):
        output1 = self.branch1(x)
        output2 = self.branch2_3(self.branch2_2(self.branch2_1(x)))
        out = torch.cat((output1, output2), dim=1)
        out = self.linear(out)
        x = self.shortcut(x)
        out=x + out * self.scale
        out=F.relu(out)
        return out
    
class ReductionA(nn.Module):
    def __init__(self,in_channel):
        super(ReductionA, self).__init__()
        self.branch1=nn.MaxPool2d(kernel_size=3,stride=2,padding=0)
        self.branch2=Conv1(in_channel=in_channel,out_channel=384,kernel_size=3,stride=2,padding=0)
        self.branch3_1=Conv1(in_channel=in_channel,out_channel=256,kernel_size=1,stride=1,padding=0)
        self.branch3_2=Conv1(in_channel=256,out_channel=256,kernel_size=3,stride=1,padding=1)
        self.branch3_3=Conv1(in_channel=256,out_channel=384,kernel_size=3,stride=2,padding=0)
    def forward(self,x):
        out1=self.branch1(x)
        out2=self.branch2(x)
        out3=self.branch3_3(self.branch3_2(self.branch3_1(x)))
        return torch.cat((out1,out2,out3),dim=1)
    
class ReductionB(nn.Module):
    def __init__(self,in_channel):
        super(ReductionB, self).__init__()
        self.branch1=nn.MaxPool2d(kernel_size=3,stride=2,padding=0)
        self.branch2_1=Conv1(in_channel=in_channel,out_channel=256,kernel_size=1,stride=1,padding=0)
        self.branch2_2=Conv1(in_channel=256,out_channel=384,kernel_size=3,stride=2,padding=0)
        self.branch3_1=Conv1(in_channel=in_channel,out_channel=256,kernel_size=1,stride=1,padding=0)
        self.branch3_2=Conv1(in_channel=256,out_channel=288,kernel_size=3,stride=2,padding=0)
        self.branch4_1=Conv1(in_channel=in_channel,out_channel=256,kernel_size=1,stride=1,padding=0)
        self.branch4_2=Conv1(in_channel=256,out_channel=288,kernel_size=3,stride=1,padding=1)
        self.branch4_3=Conv1(in_channel=288,out_channel=320,kernel_size=3,stride=2,padding=0)
    def forward(self,x):
        out1=self.branch1(x)
        out2=self.branch2_2(self.branch2_1(x))
        out3=self.branch3_2(self.branch3_1(x))
        out4=self.branch4_3(self.branch4_2(self.branch4_1(x)))
        return torch.cat((out1,out2,out3,out4),dim=1)


class InceptionResNetV2(nn.Module):
    def __init__(self,classes=6):
        super(InceptionResNetV2, self).__init__()
        blocks=[]
        blocks.append(Stem())
        for _ in range(5):
            blocks.append(InceptionResNetA(384))
        blocks.append(ReductionA(384))
        for _ in range(10):
            blocks.append(InceptionResNetB(1152))
        blocks.append(ReductionB(1152))
        for _ in range(5):
            blocks.append(InceptionResNetC(2144))
        self.map=nn.Sequential(*blocks)
        self.pool=nn.AvgPool2d(kernel_size=8)
        self.dropout=nn.Dropout(0.2)
        self.linear=nn.Linear(2144,classes)

    def forward(self,x):
        out=self.map(x)
        out=self.pool(out)
        out=self.dropout(out)
        out=self.linear(out.view(out.size(0),-1))
        return out

if __name__=="__main__":
    a=torch.randn(2,3,299,299)
    net=InceptionResNetV2()
    #print(net)
    print(net(a).shape)