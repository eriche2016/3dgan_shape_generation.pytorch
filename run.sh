#!/bin/bash

# make data 
if [ 1 -eq 0 ]; then 
    python ./scripts/convert_shapenet10.py ./data/3DShapeNets/
fi

# avoid using adam to training network 
if [ 1 -eq 0 ]; then 
    python main_d3gan.py --data_dir ./data/train/chair/ \
                        --adam \
                        --batch_size  100 \
                        --cube_len 32 \
                        --nz 200 \
                        --cuda \
                        --ngpu 1\
                        --gpu_id 1 
fi

# train wassersetin gan 
if [ 1 -eq 0 ]; then 
    python main_d3gan_wgan.py --data_dir ./data/train/chair/ \
                        --max_epochs 1500 \
                        --batch_size  100 \
                        --cube_len 32 \
                        --nz 200 \
                        --cuda \
                        --ngpu 1\
                        --gpu_id 3 
fi


# experiment 1 
if [ 1 -eq 0 ]; then 
    python main_d3gan_wgan_gp.py --data_dir ./data/train/chair/ \
                        --adam \
                        --max_epochs 1500 \
                        --batch_size  256 \
                        --cube_len 32 \
                        --nz 200 \
                        --cuda \
                        --ngpu 1\
                        --gpu_id 3 
fi 

# experiment 2 
if [ 1 -eq 1 ]; then 
    python main_d3gan_wgan_gp.py --data_dir ./data/train/chair/ \
                        --adam \
                        --max_epochs 1500 \
                        --batch_size  256 \
                        --cube_len 32 \
                        --nz 200 \
                        --cuda \
                        --ngpu 1 \
                        --gpu_id 5 
fi

if [ 1 -eq 0 ]; then
    python plot_voxels.py
fi 
