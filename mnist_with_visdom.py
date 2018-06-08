# -*- coding: UTF-8 -*-
""" Run MNIST example and log to visdom
    Notes:
        - Visdom must be installed (pip works)
        - the Visdom server must be running at start!
    Example:
        $ python -m visdom.server -port 8097 &
        $ python mnist_with_visdom.py
"""
import numpy as np
from visdom import Visdom
from tqdm import tqdm
import torch
import torch.optim
import torchnet as tnt
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.init import kaiming_normal
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.datasets.mnist import MNIST


def get_iterator(mode):               # 获得数据
    ds = MNIST(root='./', download=True, train=mode)
    data = getattr(ds, 'train_data' if mode else 'test_data')
    labels = getattr(ds, 'train_labels' if mode else 'test_labels')
    tds = tnt.dataset.TensorDataset([data, labels])
    return tds.parallel(batch_size=128, num_workers=4, shuffle=mode)


def conv_init(ni, no, k):       # He 初始化
    return kaiming_normal(torch.Tensor(no, ni, k, k))


def linear_init(ni, no):        # He 初始化
    return kaiming_normal(torch.Tensor(no, ni))


def f(params, inputs, mode):
    o = inputs.view(inputs.size(0), 1, 28, 28)
    o = F.conv2d(o, params['conv0.weight'], params['conv0.bias'], stride=2)
    o = F.relu(o)
    o = F.conv2d(o, params['conv1.weight'], params['conv1.bias'], stride=2)
    o = F.relu(o)
    o = o.view(o.size(0), -1)
    o = F.linear(o, params['linear2.weight'], params['linear2.bias'])
    o = F.relu(o)
    o = F.linear(o, params['linear3.weight'], params['linear3.bias'])
    return o


def main(n):
    viz = Visdom()
    params = {
        'conv0.weight': conv_init(1, 50, 5), 'conv0.bias': torch.zeros(50),
        'conv1.weight': conv_init(50, 50, 5), 'conv1.bias': torch.zeros(50),
        'linear2.weight': linear_init(800, 512), 'linear2.bias': torch.zeros(512),
        'linear3.weight': linear_init(512, 10), 'linear3.bias': torch.zeros(10),
    }    # 创建参数字典 conv_init 和 linear_init 采用 He正规
    params = {k: Variable(v, requires_grad=True) for k, v in params.items()}
    # torch.autograd.Variable     Tensor 转 Variable
    if n == 1:optimizer = torch.optim.SGD(params.values(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    if n == 2:optimizer = torch.optim.Adam( params.values(), lr=0.001, betas=(0.9, 0.99))
    if n == 3:optimizer = torch.optim.RMSprop(params.values(), lr=0.01, alpha=0.9)
    # 方法：SGD
    engine = Engine()
    # Engine给训练过程提供了一个模板，该模板建立了model，DatasetIterator，Criterion和Meter之间的联系
    meter_loss = tnt.meter.AverageValueMeter()           # 用于统计任意添加的变量的方差和均值，可以用来测量平均损失等
    classerr = tnt.meter.ClassErrorMeter(accuracy=True)  # 该meter用于统计分类误差
    confusion_meter = tnt.meter.ConfusionMeter(10, normalized=True)  # 多类之间的混淆矩阵

    port = 8097  # 端口

    train_loss_logger = VisdomPlotLogger('line', port=port, opts={}, win='102')
    # 定义win，name不能在这里设置，应该在这里的opts把标签legend设置完毕:
    viz.update_window_opts(
        win='101',
        opts=dict(
            legend=['Apples', 'Pears'],
            xtickmin=0,
            xtickmax=1,
            xtickstep=0.5,
            ytickmin=0,
            ytickmax=1,
            ytickstep=0.5,
            markersymbol='cross-thin-open',
        ),
    )

    # train_loss 折线
    train_err_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'Train Class Error'})   # train_err 折线
    test_loss_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'Test Loss'})           # test_loss 折线
    test_err_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'Test Class Error'},)    # test_err 折线
    confusion_logger = VisdomLogger('heatmap', port=port, opts={'title': 'Confusion matrix',
                                                                'columnnames': list(range(10)),
                                                                'rownames': list(range(10))})
    # 误判信息

    def h(sample):  # 数据获取， f(参数，输入，mode), o为结果
        inputs = Variable(sample[0].float() / 255.0)
        targets = Variable(torch.LongTensor(sample[1]))
        o = f(params, inputs, sample[2])
        return F.cross_entropy(o, targets), o     # 返回Loss，o

    def reset_meters(): # meter重置
        classerr.reset()
        meter_loss.reset()
        confusion_meter.reset()

    # hooks = {
    # ['on_start'] = function() end, --用于训练开始前的设置和初始化
    # ['on_start_epoch'] = function()end, -- 每一个epoch前的操作
    # ['on_sample'] = function()end, -- 每次采样一个样本之后的操作
    # ['on_forward'] = function()end, -- 在model: forward()之后的操作
    # ?['onForwardCriterion'] = function()end, -- 前向计算损失函数之后的操作
    # ?['onBackwardCriterion'] = function()end, -- 反向计算损失误差之后的操作
    # ['on_backward'] = function()end, -- 反向传递误差之后的操作
    # ['on_update'] = function()end, -- 权重参数更新之后的操作
    # ['on_end_epoch'] = function()end, -- 每一个epoch结束时的操作
    # ['on_end'] = function()end, -- 整个训练过程结束后的收拾现场
    # }

    # state = {
    # ['network'] = network, --设置了model
    # ['criterion'] = criterion, -- 设置损失函数
    # ['iterator'] = iterator, -- 数据迭代器
    # ['lr'] = lr, -- 学习率
    # ['lrcriterion'] = lrcriterion, --
    # ['maxepoch'] = maxepoch, --最大epoch数
    # ['sample'] = {}, -- 当前采集的样本，可以在onSample中通过该阈值查看采样样本
    # ['epoch'] = 0, -- 当前的epoch
    # ['t'] = 0, -- 已经训练样本的个数
    # ['training'] = true - - 训练过程
    # }

    # def train(self, network, iterator, maxepoch, optimizer):
    # state = {
    #      'network': network,
    #      'iterator': iterator,
    #      'maxepoch': maxepoch,
    #      'optimizer': optimizer,
    #      'epoch': 0,      # epoch
    #      't': 0,          # sample
    #      'train': True,
    #     }
    def on_sample(state):  # 每次采样一个样本之后的操作
        state['sample'].append(state['train'])    # 样本采集之后训练
        if state.get('epoch') != None and state['t'] >10:
            if n == 1 :
                train_loss_logger.log(state['t'], meter_loss.value()[0],name="SGD")
            if n == 2 :
                train_loss_logger.log(state['t'], meter_loss.value()[0],name="Adam")
            if n == 3 :
                train_loss_logger.log(state['t'], meter_loss.value()[0],name="RMSprop")
        reset_meters()

    def on_forward(state): # 在model: forward()之后的操作
        classerr.add(state['output'].data,torch.LongTensor(state['sample'][1]))
        confusion_meter.add(state['output'].data,torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data[0])

    def on_start_epoch(state): # 每一个epoch前的操作
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(state):   # 每一个epoch结束时的操作
        print('Training loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classerr.value()[0]))
        # train_loss_logger.log(state['epoch'], meter_loss.value()[0])
        # train_err_logger.log(state['epoch'], classerr.value()[0])

        # do validation at the end of each epoch
        reset_meters()
        engine.test(h, get_iterator(False))
        # test_loss_logger.log(state['epoch'], meter_loss.value()[0])
        # test_err_logger.log(state['epoch'], classerr.value()[0])
        # confusion_logger.log(confusion_meter.value())
        print('Testing loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classerr.value()[0]))

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.train(h, get_iterator(True), maxepoch=1, optimizer=optimizer)


if __name__ == '__main__':
    main(1)
    main(2)
    main(3)
