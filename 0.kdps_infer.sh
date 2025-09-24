#!/bin/bash
######################################################################
#0.KDPS_INFER --- 

# Author: Ann <xxxxxxxx@xxxxxxxxxx.xxxxx>
# Copyright © 2025, ANN, all rights reserved.
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
