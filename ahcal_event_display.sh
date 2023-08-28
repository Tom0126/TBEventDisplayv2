#!/bin/bash

source /hpcfs/cepc/higgsgpu/siyuansong/conda.env
conda activate pytorch

file_path=/hpcfs/cepc/higgsgpu/siyuansong/PID/data/SPS_2023/mu-/normal/AHCAL_Run27_20230426_015134.root
save_dir=./Result
entry_start=800
entry_end=801
random_num=1

pid=0

python ahcal.py --file_path $file_path --save_dir $save_dir --entry_start $entry_start  --pid $pid --random_num $random_num --entry_end $entry_end
