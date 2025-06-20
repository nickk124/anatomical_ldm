# Anatomical Latent Diffusion Models

This is an implementation of **Anatomical Latent Diffusion Models** that learns anatomical structure from segmentation masks during training to improve medical image generation quality, without requiring masks at inference time.

## Key Features

- **Unconditional generation** with learned anatomical awareness
- **Anatomical register bank** that captures structural relationships
- **Progressive training** with adaptive anatomical supervision
- **Compatible** with existing CompVis LDM infrastructure
- **General framework** for any medical dataset with masks

## Architecture Overview

### Anatomical Register Bank
- Learnable anatomical registers for each segmentation class
- Cross-attention mechanism provides structural context to UNet
- Spatial and temporal awareness for different generation stages

### Anatomical UNet
- Extends CompVis UNetModel with anatomical cross-attention
- Anatomical prediction head for training supervision
- Maintains full backward compatibility

### Training Strategy
1. **Training**: Uses masks for anatomical supervision + standard diffusion loss
2. **Inference**: Anatomical registers provide learned context automatically
3. **Progressive training**: Strong→balanced→minimal supervision schedule

## Installation

1. Clone and setup CompVis latent-diffusion environment
2. Add the anatomical modules to the CompVis codebase
3. Install dependencies:
   ```bash
   pip install torch torchvision pytorch-lightning
   pip install diffusers transformers
   pip install einops omegaconf
   ```

## Usage

### Training

1. **Prepare data**: Organize images and segmentation masks in the CompVis format
2. **Train VAE** (if needed): Use existing autoencoder training scripts
3. **Train Anatomical LDM**:
   ```bash
   # For CT-Organ dataset
   CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/latent-diffusion/anatomical_ctorgan_kl.yaml -t --gpus 0,1
   
   # For Breast MRI dataset  
   CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/latent-diffusion/anatomical_breastmri_kl.yaml -t --gpus 0,1
   ```

### Inference

Anatomical LDMs generate improved images without requiring masks:

```python
from ldm.models.diffusion.anatomical_ddpm import AnatomicalLatentDiffusion
from omegaconf import OmegaConf

# Load model
config = OmegaConf.load("configs/latent-diffusion/anatomical_ctorgan_kl.yaml")
model = AnatomicalLatentDiffusion(**config.model.params)
model.load_state_dict(torch.load("path/to/checkpoint.ckpt"))

# Generate samples (no masks needed!)
samples = model.sample(batch_size=8)
images = model.decode_first_stage(samples)
```

## Configuration

Key parameters in config files:

```yaml
# Anatomical-specific parameters
anatomical_mask_key: segmentation
anatomical_loss_weight: 1.0
progressive_anatomical_training: true
anatomical_warmup_steps: 10000

# UNet anatomical parameters
use_anatomical_registers: true
num_anatomical_classes: 6  # Including background
anatomical_register_dim: 512
num_registers_per_class: 4
```

## File Structure

```
ldm/
├── modules/
│   ├── anatomical_registers.py          # Register bank implementation
│   └── diffusionmodules/
│       └── anatomical_openaimodel.py    # Anatomical UNet
├── models/diffusion/
│   └── anatomical_ddpm.py               # Anatomical LDM
├── data/
│   └── medical_images.py                # Enhanced data loader
└── configs/latent-diffusion/
    ├── anatomical_ctorgan_kl.yaml       # CT-Organ config
    └── anatomical_breastmri_kl.yaml     # Breast MRI config
```

## Comparison to Standard LDMs

| Aspect | Standard LDM | Anatomical LDM |
|--------|--------------|----------------|
| **Training** | Image reconstruction only | Image + anatomical supervision |
| **Inference** | May generate unrealistic anatomy | Anatomically-aware generation |
| **Conditioning** | External (text, class, etc.) | Internal anatomical registers |
| **Medical Focus** | General purpose | Optimized for medical images |
| **Mask Requirement** | None | Training only (not inference) |

## Training Logs

Monitor these key metrics during training:

- `train/loss`: Standard diffusion loss
- `train/anatomical_loss`: Anatomical supervision loss
- `train/anatomical_weight`: Current anatomical loss weight
- `train/weighted_anatomical_loss`: Combined anatomical contribution

## Results

Anatomical LDMs show improved:
- **Anatomical consistency**: Realistic organ shapes and positions
- **Image quality**: Better overall generation quality
- **Training efficiency**: Faster convergence with anatomical guidance
- **Robustness**: More stable training dynamics

## Citation

```bibtex
@article{anatomical_ldm_2025,
  title={Anatomical Latent Diffusion Models for Medical Image Generation},
  author={},
  journal={},
  year={2025}
}
```