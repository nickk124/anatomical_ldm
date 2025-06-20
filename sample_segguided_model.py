# thanks to https://github.com/CompVis/latent-diffusion/issues/120#issuecomment-1288277021

import torch
import numpy as np

from scripts.sample_diffusion import load_model
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from einops import rearrange

from ldm.data.medical_images import *


def ldm_cond_sample(config_path, ckpt_path, dataset, batch_size):
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    x = next(iter(dataloader))

    seg = x['segmentation']

    print(seg.shape)
    print(seg.unique())

    with torch.no_grad():
        print(seg.shape)
        seg = rearrange(seg, 'b h w c -> b c h w')
        print(seg.shape)
        condition = model.to_rgb(seg)
        print(seg.shape)
        print(seg)

        seg = seg.to('cuda').float()
        seg = model.get_learned_conditioning(seg)

        samples, _ = model.sample_log(cond=seg, batch_size=batch_size, ddim=True,
                                      ddim_steps=50, eta=1.)

        samples = model.decode_first_stage(samples)

    save_image(condition, 'cond.png')
    save_image(samples, 'sample.png')


if __name__ == '__main__':

    config_path = 'configs/latent-diffusion/breastmri_maskguided_kl.yaml'
    ckpt_path = 'logs/2024-04-26T14-52-25_breastmri_maskguided_kl/checkpoints/last.ckpt'

    dataset = BreastMRITest(size=256)

    ldm_cond_sample(config_path, ckpt_path, dataset, 4)

