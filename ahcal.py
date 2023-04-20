#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/21 00:08
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : ahcal.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from ANN.net import LeNet_bn as lenet
from Data.read_root import prepare_npy
import torch

class Display():
    '''
    for pid datasets, .npy only.
    '''
    def __init__(self,file_path, tree_name, entry_start, entry_end, exps):
        self.file_path=file_path

        self.dataset = prepare_npy(file_path=file_path,tree_name=tree_name,entry_start=entry_start,entry_end=entry_end,exps=exps)
        self.predicted=None
        self.pid_flag=False
        self.num=entry_end-entry_start

    def pid(self,threshold, n_classes, model_path):
        '''
           0: mu
           1: e+
           2: pi+
           3: noise

           :param data:
           :return:
           '''

        id_dict = {
            0: 'mu+',
            1: 'e+',
            2: 'pi+',
            3: 'noise',
            -1: 'uncertain'
        }

        inputs=np.transpose(self.dataset, axes=[0, 3, 1, 2])
        inputs = torch.from_numpy(inputs)
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
            max_prb = max_prb.cpu().numpy()

            predicted[max_prb < threshold] = -1

        self.predicted=predicted
        self.pid_flag=True

    def plot(self,index,save_path):

        label_dict={
            0:'mu+',
            1:'e+',
            2:'pi+',
            3:'noise',
            -1:'uncertain'

        }
        event=self.dataset[index]

        max_ed = abs(np.amax(event))+0.0001

        fig=plt.figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(projection='3d')
        plt.gca().set_box_aspect((1, 2, 1))

        tags=np.where(event!=0)

        length_x=len(tags[0])
        assert length_x == len(tags[1])
        assert length_x == len(tags[2])

        for i in range(length_x):
            x_index=tags[0][i]
            z_index=tags[1][i]
            y_index=tags[2][i]
            x2 = np.arange(x_index, x_index + 2)
            z2 = np.arange(z_index, z_index + 2)
            x2, z2 = np.meshgrid(x2, z2)

            y2 = np.ones(x2.shape) * (y_index)

            surf2 = ax.plot_surface(x2, y2, z2, alpha=0.8, linewidth=0.1,
                                    antialiased=False, rstride=1, cstride=1,
                                    color=((1 - abs(event[x_index, z_index, y_index] / max_ed)) ** 2
                                           , (1 - abs(event[x_index, z_index, y_index] / max_ed)) ** 2
                                           , 1)
                                )
        ax.view_init(30,-40)
        ax.grid(False)
        ax.set_xticks(np.linspace(0,18,6))
        ax.set_zticks(np.linspace(0, 18, 6))
        ax.set_yticks(np.linspace(0, 40, 5))

        if self.pid_flag:
            label=label_dict.get(self.predicted[index])
            plt.title('ANN Predicts: {}'.format(label))

        plt.savefig(save_path.format(index))
        plt.close(fig)

    def plot_all(self,save_path):
        for index in range(self.num):
            self.plot(index,save_path)


if __name__ == '__main__':
    file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/AHCAL/HCAL_alone/pi+_V1/40GeV/AHCAL_Run58_20221021_184832.root'
    tree_name='Calib_Hit'
    entry_start=0
    entry_end=10
    exps = ['Hit_X', 'Hit_Y', 'Hit_Z', 'Hit_Energy']
    save_path='Result/{}.png'

    threshold=0.9
    n_classes=4
    model_path='/hpcfs/cepc/higgsgpu/siyuansong/TBEventDisplayv2/ANN/net.pth'

    display=Display(file_path=file_path,tree_name=tree_name,entry_start=entry_start,entry_end=entry_end,exps=exps)
    display.pid(threshold=threshold,n_classes=n_classes,model_path=model_path)
    display.plot_all(save_path=save_path)
    pass
