#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/21 00:21
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : pid.py
# @Software: PyCharm
import numpy as np

from net import LeNet_bn as lenet
import torch

def pid(data, threshold, n_classes, model_path):
    '''
    0: mu
    1: e+
    2: pi+
    3: noise

    :param data:
    :return:
    '''

    id_dict={
        0: 'mu+',
        1: 'e+',
        2: 'pi+',
        3: 'noise',
        -1: 'uncertain'
    }

    inputs=torch.from_numpy(np.transpose(data,axes=[0,3,1,2]))
    gpu = torch.cuda.is_available()
    device = 'cuda' if gpu else 'cpu'

    net = lenet(classes=n_classes)

    net = net.to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))


    with torch.no_grad():
        net.eval()

        inputs = inputs.to(device)
        outputs = net(inputs)

        prbs = torch.nn.Softmax(dim=1)(outputs)
        max_prb, predicted = torch.max(prbs, 1)

        predicted = predicted.cpu().numpy()
        max_prb= max_prb.cpu().numpy()

        predicted[max_prb<threshold]=-1

        return predicted










if __name__ == '__main__':
    pass
