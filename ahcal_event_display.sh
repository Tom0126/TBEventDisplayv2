#!/bin/bash

source /mnt2/SJTU/Song/conda_setup/conda.env
conda activate cepc_tb

file_path=/mnt2/AHCAL/PublicAna/2023/BeamAna/result/mu-/100GeV/AHCAL_Run19_20230425_131356.root
save_dir=./Result
entry_start=1000
entry_end=2000
random_num=10

pid=True

python ahcal.py --file_path $file_path --save_dir $save_dir --entry_start $entry_start --entry_end $entry_end  --pid $pid --random_num $random_num
