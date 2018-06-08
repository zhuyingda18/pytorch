# -*- coding: UTF-8 -*-
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import datetime
import torchnet

start = datetime.datetime.now()
class Net(nn.Module)    :                   # 网络模型
    def __init__(self):
        super(Net, self).__init__()       # 继承父类初始化
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)    # 卷积层1，输入1，输出10，kernel=5*5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)   # 卷积层2，输入10，输出20，kernel=5*5
        self.conv2_drop = nn.Dropout2d()                # Dropout2d
        self.fc1 = nn.Linear(320, 50)                   # 全连接3，320->50
        self.fc2 = nn.Linear(50, 10)                    # 全连接4，50->10

    def forward(self, x):               # 网络搭建
        x = F.relu(F.max_pool2d(self.conv1(x), 2))                  # 第1层 对x卷积，maxpool2*2，ReLU
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # 第2层 对x卷积，dropout，maxpool2*2，ReLU
        x = x.view(-1, 320)                             # 把张量x通过view编成1维
        x = F.relu(self.fc1(x))                         # 第三层 对x全连接，ReLU
        x = F.dropout(x, training=self.training)        # torch.nn.functional.dropout(input, p=0.5, training=False, inplace=False)
        x = self.fc2(x)                                 # 第四层 全连接
        return F.log_softmax(x, dim=1)                  # 返回之前softmax

def train(args, model, device, train_loader, optimizer, epoch):  # train函数
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    # optimizer = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    # optimizer = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


if __name__ == '__main__':
    main()

end = datetime.datetime.now()
print(end-start)