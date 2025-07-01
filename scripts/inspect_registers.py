"""
Quick inspection of anatomical registers in a trained model.
"""

import torch
import numpy as np
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import argparse


def inspect_model_registers(config_path, checkpoint_path):
    """Quick inspection of register usage."""
    
    # Load config and model
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    # Check if model has anatomical registers
    if not hasattr(model.model.diffusion_model, 'anatomical_registers'):
        print("❌ Model doesn't have anatomical registers!")
        return
    
    print("✅ Model has anatomical registers\n")
    
    # Get register module
    registers = model.model.diffusion_model.anatomical_registers
    register_bank = registers.register_bank
    
    # Basic stats
    print("📊 Register Statistics:")
    print(f"   Shape: {register_bank.shape}")
    print(f"   Total parameters: {register_bank.numel():,}")
    print(f"   Memory usage: {register_bank.numel() * 4 / 1024**2:.2f} MB")
    
    # Analyze register properties
    with torch.no_grad():
        # Norms
        norms = register_bank.norm(dim=-1)
        print(f"\n📏 Register Norms:")
        print(f"   Mean: {norms.mean():.4f}")
        print(f"   Std:  {norms.std():.4f}")
        print(f"   Min:  {norms.min():.4f}")
        print(f"   Max:  {norms.max():.4f}")
        
        # Similarity between registers
        num_classes = register_bank.shape[0]
        print(f"\n🔍 Inter-class Similarity:")
        for i in range(min(num_classes, 5)):  # Show first 5 classes
            class_regs = register_bank[i]
            # Compute cosine similarity
            normalized = class_regs / class_regs.norm(dim=-1, keepdim=True)
            similarity = torch.mm(normalized, normalized.t())
            avg_sim = similarity[torch.triu_indices(similarity.shape[0], similarity.shape[1], offset=1)].mean()
            print(f"   Class {i}: {avg_sim:.4f}")
        
        # Check if registers are being used (non-zero)
        zero_registers = (register_bank.abs().sum(dim=-1) < 1e-6).sum()
        print(f"\n⚠️  Dead registers: {zero_registers}/{register_bank.shape[0] * register_bank.shape[1]}")
        
        # Variance analysis
        print(f"\n📈 Register Diversity:")
        for i in range(min(num_classes, 5)):
            class_var = register_bank[i].var(dim=0).mean()
            print(f"   Class {i} variance: {class_var:.4f}")
    
    # Check attention modules
    print(f"\n🎯 Anatomical Attention Modules:")
    attn_count = 0
    for name, module in model.model.diffusion_model.named_modules():
        if 'anatomical_cross_attn' in name:
            attn_count += 1
            if attn_count <= 3:  # Show first 3
                print(f"   {name}")
    print(f"   Total: {attn_count} attention modules")
    
    # Training stats if available
    if hasattr(registers, 'register_usage_count'):
        print(f"\n📊 Training Statistics:")
        print(f"   Register updates: {registers.register_usage_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    inspect_model_registers(args.config, args.checkpoint)