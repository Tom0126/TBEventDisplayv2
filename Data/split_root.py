#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/27 22:41
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : split_root.py
# @Software: PyCharm
import numpy as np
import uproot
from read_root import ReadRoot

def split_root_based_on_ckv(file_path, save_path, ckv_0_signal, ckv_1_signal):
    exps=['Hit_X','Hit_Y','Hit_Z','Hit_Energy', 'Cherenkov']
    ahcal=ReadRoot(file_path=file_path, tree_name='Calib_Hit', exp=exps)

    x = ahcal.readBranch(exps[0])
    y = ahcal.readBranch(exps[1])
    z = ahcal.readBranch(exps[2])
    e = ahcal.readBranch(exps[3])
    ckv = ahcal.readBranch(exps[4])
    ckv=np.vstack(ckv)

    cut=np.logical_and(ckv[:,0]==ckv_0_signal, ckv[:,1]==ckv_1_signal)

    file = uproot.recreate(save_path)


    file['Calib_Hit'] = {

        'Hit_X': x[cut],
        'Hit_Y': y[cut],
        'Hit_Z': z[cut],
        'Hit_Energy': e[cut],
    }


if __name__ == '__main__':

    split_root_based_on_ckv(file_path='/home/songsy/TBEventDisplayv2/data/AHCAL/PublicAna/2023/BeamAna/result/PS/pi-/10GeV/AHCAL_Run112_20230527_064741.root',
                            save_path='/home/songsy/TBEventDisplayv2/data/root_file/run112_0_0.root',
                            ckv_0_signal=0,
                            ckv_1_signal=0)
    pass
