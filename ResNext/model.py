import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4
    #卷积层个数的变化
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,                     #定义初始函数及残差结构所需要使用的一系列层结构
                    groups=1, width_per_group=64):                                                #resnext多传入groups和width_per_group
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups                            #计算resnet和rennext网络第一个卷积层和第二个卷积层采用的卷积核个数

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,                     #对于resnet，不传入groups和width_per_group两个参数，out_channels=width=out_channels
                                kernel_size=1, stride=1, bias=False)  # squeeze channels        #对于resnext，传入这两个参数，width等于两倍的resnet的out_channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,           
                                kernel_size=3, stride=stride, bias=False, padding=1)            #步长为2，因此这里步长根据传入的stride调整
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,     #卷积核个数为四倍的前一层卷积核个数
                                kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=3,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #m.weight 表示要初始化的权重，mode='fan_out' 表示使用 Kaiming 初始化方法中的“扇出”方式计算标准差，nonlinearity='relu' 表示使用 ReLU 激活函数。
                #该段代码使用了 self.modules() 方法遍历神经网络中的所有模块，如果遍历到的模块是卷积层，则对其权重进行 Kaiming 初始化。这种初始化方法可以帮助神经网络更好地学习数据集中的特征和模式，从而提高模型的性能。
                #在第一个 epoch 中，模型初始权重是随机初始化的，因此模型的表现可能会比较差。此时，模型还没有学习到数据集中的特征和模式，因此准确度可能会相对较低。但是随着训练的进行，模型会逐渐学习到数据集中的特征和模式，因此准确度也会逐渐提高。
                #当进入第二个 epoch 时，模型已经学习到了一些数据集中的特征和模式。此时，模型的权重已经不再是随机初始化的，而是根据第一个 epoch 的训练结果进行了更新。因此，在第二个 epoch 中，模型可能会更好地利用已经学习到的特征和模式，从而获得更好的性能。
                #另外，如果在训练过程中使用了学习率衰减等技巧，也可能导致第二个 epoch 的性能提高。学习率衰减可以使模型在后续的 epoch 中更加稳定和鲁棒。
                #初始化是指在训练神经网络之前，对网络中的权重和偏置进行初始化的过程。在初始化过程中，会为每个参数赋予一个初始值，这些初始值通常是随机的，并且会根据不同的初始化方法进行分配。
                #初始化的目的是使神经网络在训练过程中更容易地学习到数据集中的特征和模式，并且更容易地收敛到最优解。如果网络中的参数初始值过大或过小，可能会导致梯度消失或梯度爆炸等问题，从而影响网络的训练效果和收敛速度。
                #通过合适的初始化方法，可以使网络中的参数初始值更加合理，从而提高网络的训练效果和收敛速度。例如，在使用 ReLU 激活函数时，可以使用 Kaiming 初始化方法来初始化权重，以避免梯度消失问题，同时加快网络的收敛速度。
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def resnext50_32x4d(num_classes=6, include_top=True):                   #
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth     #权重下载地址
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],                                #调用ResNet类，与之前ResNet相同，但多了两个参数
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,                                           #多传入了这两个参数
                  width_per_group=width_per_group)
