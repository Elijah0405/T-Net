import os
import sys
import json
import math
import pandas as pd

import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

from TNet import TNet

starttime = datetime.datetime.now()

def main():
    
    batch_size = 64
    epochs = 50
    lr = 0.002 
    wd = 0.0001
    #学习率的作用是主导模型参数的更新过程，它决定了每次参数更新的步长大小。在训练过程中，模型会根据损失函数计算出每个参数的梯度，然后使用梯度下降等优化算法更新模型参数。学习率就是用来控制每次参数更新的步长大小的。
    #求梯度是指计算损失函数对模型参数的导数，它用于指导模型参数的更新。在深度学习中，通常使用反向传播算法来求解梯度，从而更新模型参数。在反向传播过程中，梯度是对损失函数对每个参数的导数进行计算得到的，而不是对图片的张量进行计算得到的。

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    #根目录
    root_dir = os.getcwd()
    #绝对路径，父路径
    script_dir = os.path.dirname(__file__)
    weights_dir = script_dir + '/weights'
    weight_save_dir = weights_dir + '/epoch{}_bs{}_lr{}_wd{}.pth'.format(epochs,batch_size,lr,wd)
    log_dir = script_dir + '/log_epoch{}_bs{}_lr{}_wd{}'.format(epochs,batch_size,lr,wd)

    if os.path.exists(weights_dir) is False:
        os.makedirs(weights_dir)

    tb_writer = SummaryWriter(log_dir=log_dir)

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.539, 0.551, 0.559], [0.106, 0.104, 0.107])]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.540, 0.552, 0.560], [0.105, 0.103, 0.105])])}

    data_root = os.path.abspath(os.path.join(os.getcwd()))  # get data root path
    image_path = os.path.join(data_root, "datasets")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])

    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    jason_dir = os.path.dirname(__file__) + '/class_indices.json'
    with open(jason_dir, 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 12])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    #这里的num_worker就是训练时加载数据的线程数目，及cpu，gpu线程数目

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    net = TNet(num_classes=6)
   
    # change fc layer structure
    #in_channel = net.fc.in_features
    #net.fc = nn.Linear(in_channel, 6)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    pg = [p for p in net.parameters() if p.requires_grad]
    #optimizer = optim.Adam(params, lr=lr, weight_decay=0.001)
    #与ViT网络相比，这里使用的Adam优化器具有自适应性，即可以自动调整学习率，默认学习率是10e-3
    optimizer = optim.Adam(pg, lr=lr, weight_decay=wd)
    #两个超参数 α控制学习率,β控制指数加权平均数,β最常用的值是0.9
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - 0.01) + 0.01
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0
    for epoch in range(epochs):
        # train
        net.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        #tqdm()是一个快速，可扩展的进度条，可在循环中加入进度提醒。

        train_loss = torch.zeros(1).to(device)  # 累计损失
        train_acc = torch.zeros(1).to(device)   # 累计预测正确的样本数
        sample_num = 0
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            sample_num += images.shape[0]
            logits = net(images.to(device))
            logits_classes = torch.max(logits, dim=1)[1]
            train_acc += torch.eq(logits_classes, labels.to(device)).sum()
            loss = loss_function(logits, labels.to(device))

            loss.backward()
            train_loss += loss.detach()
            optimizer.step()

            train_bar.desc = "train epoch[{}/{}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(epoch + 1,
                                                                                   epochs,
                                                                                   train_loss.item() / (step + 1),
                                                                                   train_acc.item() / sample_num,
                                                                                   optimizer.param_groups[0]["lr"])
        scheduler.step()
            
        # validate
        net.eval()
        
        with torch.no_grad():
            accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
            accu_loss = torch.zeros(1).to(device)  # 累计损失
            val_num = 0

            val_bar = tqdm(validate_loader, file=sys.stdout)
            for step, val_data in enumerate(val_bar):
                val_images, val_labels = val_data
            
                val_num += val_images.shape[0]
                #val_images.shape[0]就是第一维度batch_size的大小

                outputs = net(val_images.to(device))
                #图片映射到设备上，输入实例化的网络，输出最终的结果。
                predict_y = torch.max(outputs, dim=1)[1]
                # torch.max() 函数返回的结果是一个元组，包含两个张量：最大值张量和最大值所在的索引张量。
                # 具体来说，当 dim=1 时，函数会对每一行进行操作，返回每一行最大值所在的列索引张量和每一行最大值组成的张量。
                # 因此，通过使用 [1] 可以获取每一行最大值所在的列索引张量。
                accu_num += torch.eq(predict_y, val_labels.to(device)).sum()

                loss = loss_function(outputs, val_labels.to(device))
                accu_loss += loss

                val_bar.desc = "valid epoch[{}/{}] loss: {:.3f}, acc: {:.3f}".format(epoch + 1,
                                                                                     epochs,
                                                                                     accu_loss.item() / (step + 1),
                                                                                     accu_num.item() / val_num)
        
        #print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              #(epoch + 1, train_loss / train_steps, val_accurate))
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss.item() / (step + 1), epoch)
        tb_writer.add_scalar(tags[1], train_acc.item() / sample_num, epoch)
        tb_writer.add_scalar(tags[2], accu_loss.item() / (step + 1), epoch)
        tb_writer.add_scalar(tags[3], accu_num.item() / val_num, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.close()

        val_accurate = accu_num.item() / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), weight_save_dir)
    

if __name__ == '__main__':
    main()
    endtime = datetime.datetime.now()
    runtime = endtime - starttime
    list = []
    list.append(runtime)
    time = pd.DataFrame(list)
    time.columns=['time']
    time_save_dir = os.path.dirname(__file__) + '/run_time.csv'
    time.to_csv(time_save_dir,mode='a',encoding='utf-8',header=True,index=True)
    print('Finished Training')
