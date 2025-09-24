#!/bin/bash
######################################################################
#1.1.PARALLEL_KDPS_VARYK --- 

# Author: Ann <xxxxxxxx@xxxxxxxxxx.xxxxx>
# Copyright © 2025, ANN, all rights reserved.
# Created: 16 九月 2025
######################################################################

######################### Commentary ##################################
##  
######################################################################

export python=${HOME}/anaconda3/envs/vser/bin/python3
export root_dir="${HOME}/vser/"
export CUDA_VISIBLE_DEVICES=0,1,4,5,6,7
export output_path="${root_dir}d_kps_vary_K___parallel/"

python KDPS_parallel.py \
	--prompts "wikipedia_mini" \
	--K 20000 \
	--batch_size 32 \
	--model "meta-llama/Llama-3.2-1B" \
	--max_tokens 32 \
	--sampling "nucleus" \
	--output_dir=${output_path}



echo "RUNNING 1.1.parallel_kdps_varyK.sh DONE."
# 1.1.parallel_kdps_varyK.sh ends here
