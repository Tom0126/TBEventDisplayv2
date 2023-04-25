#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 23:11
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : read_root.py
# @Software: PyCharm

import uproot
import numpy as np
import matplotlib.pyplot as plt

class ReadRoot():

    def __init__(self, file_path, tree_name,start=None,end=None,cut=None, exp=None):
        file = uproot.open(file_path)
        tree = file[tree_name]

        self.tree = tree.arrays(aliases=None, cut=cut, expressions=exp, library="np", entry_start=start,
						 entry_stop=end)

    def readBranch(self, branch):
        return self.tree[branch]

def prepare_npy(file_path, tree_name,entry_start, entry_end, exps):
    '''
    only one event

    '''

    root_data=ReadRoot(file_path=file_path,tree_name=tree_name,start=entry_start,end=entry_end,exp=exps)

    x = root_data.readBranch(exps[0])
    y = root_data.readBranch(exps[1])
    z = root_data.readBranch(exps[2])
    e = root_data.readBranch(exps[3])


    # read raw root file

    num_events = len(e)
    assert num_events == len(x)
    assert num_events == len(y)
    assert num_events == len(z)

    # NHWC
    deposits = np.zeros((num_events, 18, 18, 40))

    for i in range(num_events):

        energies_ = e[i]

        x_ = np.around((x[i] + 342.5491) / 40.29964).astype(int)
        y_ = np.around((y[i] + 343.05494) / 40.29964).astype(int)
        z_ = ((z[i]) / 300).astype(int)
        num_events_ = len(energies_)
        assert num_events_ == len(x_)
        assert num_events_ == len(y_)
        assert num_events_ == len(z_)

        for j in range(num_events_):
            deposits[i, x_[j], y_[j], z_[j]] += energies_[j]
    # NCHW

    return deposits.astype(np.float32)


if __name__ == '__main__':
    pass
