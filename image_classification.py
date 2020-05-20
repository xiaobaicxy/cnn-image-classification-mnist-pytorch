# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:37:08 2020
采用CNN对mnist数据集分类
@author: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=50, kernel_size=3, stride=1, padding=0, bias=False)
        self.fc1 = nn.Linear(in_features=5*5*50, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=10)
    def forward(self, x):
        #x [1, 28, 28]
        x = F.relu(self.conv1(x)) #[100, 26, 26]
        x = F.max_pool2d(x, 2, 2) #[100, 13, 13]
        x = F.relu(self.conv2(x)) #[50, 11, 11]
        x = F.max_pool2d(x, 2, 2) #[50, 5, 5]
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, train_loader, optimizer, epoch):
    model.train()
    for data, label in train_loader:
        optimizer.zero_grad()
        pred = model(data)
        loss = F.nll_loss(pred, label)
        loss.backward()
        optimizer.step()
    print("loss:", loss.item())

def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad(): #验证不需要梯度
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        print("acc:", 100 * correct / len(test_loader.dataset))
            

#预处理数据
#mnist图片大小为[1,28,28]
batch_size = 64
torch.manual_seed(100)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)) #对每一个通道进行norm
                   ])),
    batch_size=batch_size, shuffle=True)


lr = 1e-2
momentum = 0.5
model = Net()


epochs = 1
isTrain = False
if(isTrain):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for epoch in range(epochs):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)

torch.save(model.state_dict(),"mnist_cnn0.pt")
#torch.save(model, "minist_cnn1.pt")

isTest = True
if(isTest):
    model.load_state_dict(torch.load("mnist_cnn0.pt"))
    test(model, test_loader)
    #themodel = torch.load("minist_cnn1.pt")
    #test(themodel, test_loader)
    