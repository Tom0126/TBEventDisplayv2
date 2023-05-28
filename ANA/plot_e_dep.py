#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/27 23:07
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : plot_e_dep.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from read_root import ReadRoot

def plot_e_dep(file_path):
    exp=['Hit_Energy']
    e_dep=ReadRoot(file_path=file_path, exp=exp, tree_name='Calib_Hit')
    e=e_dep.readBranch(exp[0])

    e_dep=[]

    for _ in e:

        e_dep.append(np.sum(_))

    plt.figure(figsize=(6,5))
    plt.hist(e_dep, bins=100)

    plt.show()


if __name__ == '__main__':

    plot_e_dep(
        file_path='/home/songsy/TBEventDisplayv2/data/root_file/run123_0_0.root'
    )
    pass
