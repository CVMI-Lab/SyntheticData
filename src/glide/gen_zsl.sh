#!/bin/sh
#  sh glide/gen_zsl.sh /path/to/save/dataset.pkl /path/to/save/dataset
#  sh glide/gen_zsl.sh ../dataset_eurosat.pkl syn_eurosat

pkl_path=$1
save_path=$2

for i in $(seq 0 1 7)
do
  CUDA_VISIBLE_DEVICES=${i} python3.7 glide/glide_zsl.py ${i} ${pkl_path} ${save_path} &
done






