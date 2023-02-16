#!/bin/sh
#  sh glide/gen_fsl.sh /path/to/few-shot/images /path/to/save/dataset

ref_img_path=$1
save_path=$2

for i in $(seq 0 1 7)
do
  CUDA_VISIBLE_DEVICES=${i} python3.7 glide/glide_fsl.py ${i} ${ref_img_path} ${save_path} &
done






