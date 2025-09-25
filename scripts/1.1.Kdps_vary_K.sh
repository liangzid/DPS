#!/bin/bash
######################################################################
#1.1.KDPS_VARY_K --- 

# Author: Ann <xxxxxxxx@xxxxxxxxxx.xxxxx>
# Copyright © 2025, ANN, all rights reserved.
######################################################################

######################### Commentary ##################################
##  
######################################################################

export root_dir="${HOME}/vser/"
export CUDA_VISIBLE_DEVICES=3
export output_path="${root_dir}d_kps_vary_K"

python KDPS.py \
	--prompts "wikipedia_mini" \
	--K 12000 \
	--batch_size 32 \
	--model "meta-llama/Llama-3.2-1B" \
	--max_tokens 30 \
	--sampling "nucleus" \
	--output_dir=${output_path}


echo "RUNNING 1.1.Kdps_vary_K.sh DONE."
# 1.1.Kdps_vary_K.sh ends here
