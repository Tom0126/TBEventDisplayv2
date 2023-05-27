#!/bin/bash

source /mnt2/SJTU/Song/conda_setup/conda.env
conda activate cepc_tb

file_path=/home/songsy/TBEventDisplayv2/data/root_file/run112_0_0.root
save_dir=./Result
entry_start=0
entry_end=None
random_num=100

pid=True

python ahcal.py --file_path $file_path --save_dir $save_dir --entry_start $entry_start  --pid $pid --random_num $random_num
