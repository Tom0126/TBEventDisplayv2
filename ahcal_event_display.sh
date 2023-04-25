#!/bin/bash

source ~/conda.env
conda activate pytorch

file_path=/hpcfs/cepc/higgsgpu/siyuansong/PID/data/AHCAL/HCAL_alone/pi+_V1/20GeV/AHCAL_Run100_20221023_080000.root
save_dir=./Result
entry_start=0
entry_end=10

pid=True

python ahcal.py --file_path $file_path --save_dir $save_dir --entry_start $entry_start --entry_end $entry_end  --pid $pid
