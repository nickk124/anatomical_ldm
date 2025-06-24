import argparse
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

def load_model_from_config(config, ckpt_path):
    print(f"Loading model from {ckpt_path}")
    model = instantiate_from_config(config.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.cuda()
    model.eval()
    return model

def sample_unconditional(model, n_samples, batch_size, ddim_steps, ddim_eta, output_dir):
    """Sample images from unconditional diffusion model"""
    os.makedirs(output_dir, exist_ok=True)
    
    sampler = DDIMSampler(model)
    
    # Get the shape for sampling
    shape = [model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]
    
    all_samples = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        with model.ema_scope("Sampling"):
            for i in tqdm(range(n_batches), desc="Sampling batches"):
                current_batch_size = min(batch_size, n_samples - i * batch_size)
                
                # Sample latents
                samples, _ = sampler.sample(S=ddim_steps,
                                          batch_size=current_batch_size,
                                          shape=shape,
                                          verbose=False,
                                          eta=ddim_eta)
                
                # Decode latents to images
                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                
                # Save individual images
                for j, x_sample in enumerate(x_samples):
                    x_sample = 255. * x_sample.cpu().numpy().transpose(1, 2, 0)
                    img = Image.fromarray(x_sample.astype(np.uint8).squeeze())
                    img.save(os.path.join(output_dir, f"sample_{i*batch_size+j:05d}.png"))
                
                all_samples.append(x_samples.cpu())
    
    print(f"Saved {n_samples} samples to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of DDIM steps")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta (0=deterministic)")
    parser.add_argument("--output_dir", type=str, default="samples", help="Output directory")
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Load model
    model = load_model_from_config(config, args.ckpt)
    
    # Sample
    sample_unconditional(model, args.n_samples, args.batch_size, 
                        args.ddim_steps, args.ddim_eta, args.output_dir)

if __name__ == "__main__":
    main()