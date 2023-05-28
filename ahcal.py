#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/21 00:08
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : ahcal.py
# @Software: PyCharm
import os.path

import numpy as np
import matplotlib.pyplot as plt
from ANN.net import LeNet_bn as lenet
from Data.read_root import prepare_npy
import torch
import argparse

class Display():
    '''
    for pid datasets, .npy only.
    '''
    def __init__(self,file_path, tree_name, exps, entry_start=None, entry_end=None, random_num=None):
        self.file_path=file_path

        self.dataset = prepare_npy(file_path=file_path,tree_name=tree_name,entry_start=entry_start,entry_end=entry_end,exps=exps)
        self.predicted=None
        self.pid_flag=False
        num_data = len(self.dataset)
        self.choices=np.arange(num_data) if random_num==None else np.random.choice(np.arange(num_data), random_num,replace=False)
        self.entry_start = 0 if entry_end == None else entry_start

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
            0: 'mu',
            1: 'e',
            2: 'pion',
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

    def plot(self,index,save_dir):

        label_dict={
            0:'mu',
            1:'e',
            2:'pion',
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
                                    color=((1 - abs(event[x_index, z_index, y_index] / max_ed)) ** 100
                                           , (1 - abs(event[x_index, z_index, y_index] / max_ed)) ** 100
                                           , 1)
                                )
        ax.view_init(30,-40)
        ax.grid(False)
        ax.set_xticks(np.linspace(0,18,6))
        ax.set_zticks(np.linspace(0, 18, 6))
        ax.set_yticks(np.linspace(0, 40, 5))
        ax.text2D(0.05, 0.95, "Test Beam", transform=ax.transAxes, fontsize=15, fontstyle='oblique',
                  fontweight='bold', )
        ax.text2D(0.05, 0.9, "AHCAL E_Dep @{} MeV".format(round(np.sum(event))), transform=ax.transAxes, fontsize=10,)
        if self.pid_flag:
            label=label_dict.get(self.predicted[index])
            ax.text2D(0.05, 0.85, 'ANN Predicts: {}'.format(label), transform=ax.transAxes,
                      fontsize=10, )


        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path=os.path.join(save_dir, '{}.png'.format(self.entry_start+index))

        plt.savefig(save_path)
        plt.close(fig)

    def plot_all(self,save_dir):
        for index in self.choices:
            self.plot(index,save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # base setting
    parser.add_argument("--tree_name",default='Calib_Hit', type=str, help="the tree name to read")
    parser.add_argument("--file_path", type=str, help="root file.")
    parser.add_argument("--save_dir", type=str, help="the directory to save pictures.")
    parser.add_argument("--entry_start", type=int, default=0, help="entry start.")
    parser.add_argument("--entry_end", type=int, default=None, help="entry end.")
    parser.add_argument("--random_num", type=int,default=None, help="random picked entry to event display.")
    parser.add_argument("--n_classes", type=int, default=4, help="class numbers.")
    parser.add_argument("--pid", type=bool, help="if use ANN PID tool to predict the incident particle.")
    parser.add_argument("--threshold", type=float,default=0.9, help="ANN threshold.")
    args = parser.parse_args()

    file_path=args.file_path
    tree_name=args.tree_name
    entry_start=args.entry_start
    entry_end=args.entry_end
    random_num=args.random_num


    exps = ['Hit_X', 'Hit_Y', 'Hit_Z', 'Hit_Energy']
    save_dir=args.save_dir


    threshold=args.threshold
    n_classes=args.n_classes

    model_path='./ANN/net.pth'

    display=Display(file_path=file_path,tree_name=tree_name,entry_start=entry_start,entry_end=entry_end,exps=exps, random_num=random_num)

    if args.pid:
        display.pid(threshold=threshold,n_classes=n_classes,model_path=model_path)

    display.plot_all(save_dir=save_dir)
    pass
