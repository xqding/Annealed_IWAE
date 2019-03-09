#!/bin/bash

num_samples=50
for hidden_size in 50; do
    id=$(bash $HOME/scripts/check_gpu_id.sh)
    while [ ! $id ]; do
	sleep 2
	id=$(bash $HOME/scripts/check_gpu_id.sh)
    done    
    export CUDA_VISIBLE_DEVICES=$id
    python ./script/AIWAE_train.py \
	   --hidden_size $hidden_size \
	   --num_samples $num_samples \
	&> ./log/AIWAE_hidden_size_${hidden_size}_num_samples_${num_samples}.log
    sleep 10
done
