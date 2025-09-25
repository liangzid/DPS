#!/bin/bash
######################################################################
#2.2.COMPUTE_DRAW_CONCENTRATION --- 

# Author: Ann <xxxxxxxx@xxxxxxxxxx.xxxxx>
# Copyright © 2025, ANN, all rights reserved.
######################################################################

######################### Commentary ##################################
##  
######################################################################


export python=${HOME}/anaconda3/envs/vser/bin/python3
export root_dir="${HOME}/vser/"
export CUDA_VISIBLE_DEVICES=0,1,4,5,6,7
export input_path="${root_dir}d_kps_vary_K___parallel/"

python draw_concentration.py \
	--json_file "${input_path}k_dps_output_20250917_144900.json" \
	--K_step 50\
	--N_repeats 5




echo "RUNNING 2.2.compute_draw_concentration.sh DONE."
# 2.2.compute_draw_concentration.sh ends here
