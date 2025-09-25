#!/bin/bash
######################################################################
#1.2.PARALLEL_KDPS_DIFFERENT_DATASET --- 

# Author: Ann <xxxxxxxx@xxxxxxxxxx.xxxxx>
# Copyright © 2025, ANN, all rights reserved.
######################################################################

######################### Commentary ##################################
##  
######################################################################


export python=${HOME}/anaconda3/envs/vser/bin/python3
export root_dir="${HOME}/vser/"
export CUDA_VISIBLE_DEVICES=4,5,6,7

export dataset_ls=("wikipedia_mini" "tulu-3-sft" "openo1-sft" "hh-rlhf" "alpaca")

for dataset in ${dataset_ls[*]}
do
    export output_path="${root_dir}k_dps__parallel_dataset__${dataset}/"
    python KDPS_parallel.py \
	    --prompts $dataset \
	    --K 2500 \
	    --batch_size 32 \
	    --model "meta-llama/Llama-3.2-1B" \
	    --max_tokens 32 \
	    --sampling "nucleus" \
	    --output_dir=${output_path}
done


echo "RUNNING 1.2.parallel_kdps_different_dataset.sh DONE."
# 1.2.parallel_kdps_different_dataset.sh ends here
