#!/bin/bash

# Training script for Anatomical Latent Diffusion Model
# Choose dataset config below

# For CT-Organ dataset
CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/latent-diffusion/anatomical_ctorgan_kl.yaml -t --gpus 0,1

# For Breast MRI dataset (uncomment to use)
# CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/latent-diffusion/anatomical_breastmri_kl.yaml -t --gpus 0,1