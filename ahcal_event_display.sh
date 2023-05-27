#!/bin/bash

source /mnt2/SJTU/Song/conda_setup/conda.env
conda activate cepc_tb

file_path=/home/songsy/TBEventDisplayv2/data/AHCAL/PublicAna/2023/BeamAna/result/PS/pi-/10GeV/AHCAL_Run112_20230527_064741.root
save_dir=./Result
entry_start=0
entry_end=None
random_num=100

pid=True

python ahcal.py --file_path $file_path --save_dir $save_dir --entry_start $entry_start  --pid $pid --random_num $random_num
