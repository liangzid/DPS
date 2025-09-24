#!/bin/bash
######################################################################
#1.3.PARALLEL_KDPS_DIFFERENT_MODEL --- 

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2025, ZiLiang, all rights reserved.
# Created: 22 九月 2025
######################################################################

######################### Commentary ##################################
##  
######################################################################


export python=${HOME}/anaconda3/envs/vser/bin/python3
export root_dir="${HOME}/vser/"
export CUDA_VISIBLE_DEVICES=2,3,4,5

export dataset="wikipedia_mini"
export model_ls=(
    "meta-llama/Llama-3.2-1B"\
    "Qwen/Qwen3-8B"\
    "meta-llama/Meta-Llama-3-8B-Instruct"\
    "Qwen/Qwen3-32B")

for model in ${model_ls[*]}
do
    echo "============================================================"
    echo "Current Model: $model"
    echo "============================================================"
    export output_path="${root_dir}k_dps__parallel_model__${model}/"
    python KDPS_parallel.py \
	    --prompts ${dataset} \
	    --K 2500 \
	    --batch_size 32 \
	    --model ${model} \
	    --max_tokens 32 \
	    --sampling "nucleus" \
	    --output_dir=${output_path}
done



echo "RUNNING 1.3.parallel_kdps_different_model.sh DONE."
# 1.3.parallel_kdps_different_model.sh ends here
