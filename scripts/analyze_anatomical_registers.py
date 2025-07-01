"""
Analyze and visualize anatomical register usage in trained models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import argparse
from einops import rearrange
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os


def load_model(config_path, checkpoint_path):
    """Load trained anatomical LDM model."""
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    model.cuda()
    
    return model


def analyze_register_weights(model):
    """Analyze the learned register weights and their statistics."""
    if not hasattr(model.model.diffusion_model, 'anatomical_registers'):
        print("Model doesn't have anatomical registers!")
        return
    
    registers = model.model.diffusion_model.anatomical_registers
    
    # Get register bank weights
    register_bank = registers.register_bank.detach().cpu().numpy()
    num_classes, num_registers, dim = register_bank.shape
    
    print(f"\nRegister Bank Shape: {register_bank.shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Registers per class: {num_registers}")
    print(f"Register dimension: {dim}")
    
    # Analyze register statistics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Register norms by class
    register_norms = np.linalg.norm(register_bank, axis=2)
    ax = axes[0, 0]
    sns.heatmap(register_norms, ax=ax, cmap='viridis')
    ax.set_title('Register Norms by Class')
    ax.set_xlabel('Register Index')
    ax.set_ylabel('Class')
    
    # 2. Inter-register similarity within classes
    ax = axes[0, 1]
    similarities = []
    for c in range(num_classes):
        class_registers = register_bank[c]
        sim_matrix = np.corrcoef(class_registers)
        similarities.append(sim_matrix)
    
    avg_similarity = np.mean([np.mean(s[np.triu_indices_from(s, k=1)]) for s in similarities])
    ax.bar(range(num_classes), [np.mean(s[np.triu_indices_from(s, k=1)]) for s in similarities])
    ax.set_title(f'Avg Inter-Register Similarity by Class (Overall: {avg_similarity:.3f})')
    ax.set_xlabel('Class')
    ax.set_ylabel('Average Correlation')
    
    # 3. Register activation patterns
    ax = axes[1, 0]
    register_stds = np.std(register_bank, axis=2)
    sns.heatmap(register_stds, ax=ax, cmap='coolwarm')
    ax.set_title('Register Activation Variance by Class')
    ax.set_xlabel('Register Index')
    ax.set_ylabel('Class')
    
    # 4. PCA visualization of registers
    ax = axes[1, 1]
    all_registers = register_bank.reshape(-1, dim)
    pca = PCA(n_components=2)
    registers_pca = pca.fit_transform(all_registers)
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    for c in range(num_classes):
        start_idx = c * num_registers
        end_idx = (c + 1) * num_registers
        ax.scatter(registers_pca[start_idx:end_idx, 0], 
                  registers_pca[start_idx:end_idx, 1],
                  c=[colors[c]], label=f'Class {c}', s=100)
    
    ax.set_title(f'PCA of Anatomical Registers (explained var: {sum(pca.explained_variance_ratio_):.2%})')
    ax.legend()
    
    plt.tight_layout()
    return fig


def compare_with_without_registers(model, num_samples=8):
    """Compare generation with and without anatomical registers."""
    # Generate with registers
    model.model.diffusion_model.use_anatomical_registers = True
    with torch.no_grad():
        samples_with, _ = model.sample_log(
            cond=None,
            batch_size=num_samples,
            ddim=True,
            ddim_steps=200,
            eta=1.0
        )
    
    # Generate without registers (if possible to disable)
    original_use_registers = model.model.diffusion_model.use_anatomical_registers
    model.model.diffusion_model.use_anatomical_registers = False
    with torch.no_grad():
        samples_without, _ = model.sample_log(
            cond=None,
            batch_size=num_samples,
            ddim=True,
            ddim_steps=200,
            eta=1.0
        )
    
    # Reset
    model.model.diffusion_model.use_anatomical_registers = original_use_registers
    
    # Visualize comparison
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    
    for i in range(num_samples):
        # With registers
        img_with = model.decode_first_stage(samples_with[i:i+1])[0, 0].cpu().numpy()
        axes[0, i].imshow(img_with, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('With Registers', fontsize=12)
        
        # Without registers
        img_without = model.decode_first_stage(samples_without[i:i+1])[0, 0].cpu().numpy()
        axes[1, i].imshow(img_without, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Without Registers', fontsize=12)
    
    plt.suptitle('Generation With vs Without Anatomical Registers')
    return fig


def visualize_register_usage_over_timesteps(model, num_samples=4):
    """Visualize how register attention changes over denoising timesteps."""
    model.eval()
    
    # Hook to capture attention weights
    attention_weights = []
    
    def attention_hook(module, input, output):
        # This would capture attention weights if accessible
        pass
    
    # Generate samples with timestep tracking
    with torch.no_grad():
        samples, intermediates = model.sample_log(
            cond=None,
            batch_size=num_samples,
            ddim=True,
            ddim_steps=50,
            eta=1.0
        )
    
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to model config')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='register_analysis', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=8, help='Number of samples to generate')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(args.config, args.checkpoint)
    
    # 1. Analyze register weights
    print("\nAnalyzing register weights...")
    fig = analyze_register_weights(model)
    if fig:
        fig.savefig(os.path.join(args.output_dir, 'register_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # 2. Compare with/without registers
    print("\nComparing generation with/without registers...")
    fig = compare_with_without_registers(model, num_samples=args.num_samples)
    if fig:
        fig.savefig(os.path.join(args.output_dir, 'register_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # 3. Generate samples for visual inspection
    print("\nGenerating samples...")
    samples = visualize_register_usage_over_timesteps(model, num_samples=args.num_samples)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()