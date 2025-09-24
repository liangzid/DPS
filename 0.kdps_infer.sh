#!/bin/bash
######################################################################
#0.KDPS_INFER --- 

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2025, ZiLiang, all rights reserved.
# Created: 16 九月 2025
######################################################################

######################### Commentary ##################################
##  
######################################################################

# export python=${HOME}/anaconda3/envs/vser/bin/python3
export root_dir="${HOME}/vser/"
export CUDA_VISIBLE_DEVICES=2
export output_path="${root_dir}d_kps_outputs"

python KDPS.py \
	--prompts "wikipedia_mini" \
	--K 100 \
	--batch_size 16 \
	--model "Qwen/Qwen3-4B" \
	--max_tokens 20 \
	--sampling "nucleus" \
	--output_dir=${output_path}




echo "RUNNING 0.kdps_infer.sh DONE."
# 0.kdps_infer.sh ends here
