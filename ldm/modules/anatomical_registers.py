"""
Anatomical Register Bank for medical image generation.
Learns anatomical representations from segmentation masks during training.
Provides anatomical context during generation without requiring masks at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np


class AnatomicalRegisterBank(nn.Module):
    """
    Learnable anatomical registers that provide structural context for diffusion models.
    
    During training: Supervised by segmentation masks to learn anatomical representations
    During inference: Provides learned anatomical context without requiring masks
    """
    
    def __init__(
        self,
        num_classes,
        register_dim=512,
        num_registers_per_class=4,
        spatial_dim=64,  # Expected spatial dimension of latent features
        use_spatial_encoding=True,
        dropout=0.1,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.register_dim = register_dim
        self.num_registers_per_class = num_registers_per_class
        self.total_registers = num_classes * num_registers_per_class
        self.spatial_dim = spatial_dim
        self.use_spatial_encoding = use_spatial_encoding
        
        # Learnable anatomical registers
        # Shape: [num_classes, num_registers_per_class, register_dim]
        self.registers = nn.Parameter(
            torch.randn(num_classes, num_registers_per_class, register_dim) * 0.02
        )
        
        # Spatial position encoding (optional)
        if use_spatial_encoding:
            self.spatial_encoder = SpatialPositionEncoding(register_dim, spatial_dim)
        
        # Cross-attention projection layers
        self.to_q = nn.Linear(register_dim, register_dim)
        self.to_k = nn.Linear(register_dim, register_dim)
        self.to_v = nn.Linear(register_dim, register_dim)
        
        # Anatomical prediction head (for training supervision)
        self.anatomical_predictor = nn.Sequential(
            nn.Conv2d(register_dim, register_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(register_dim // 2, num_classes, 1)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm
        self.norm = nn.LayerNorm(register_dim)
    
    def get_registers(self, batch_size, device, timestep=None):
        """
        Get anatomical registers for the given batch.
        
        Args:
            batch_size: Number of samples in batch
            device: Device to place registers on
            timestep: Optional timestep for time-aware register selection
            
        Returns:
            registers: [batch_size, total_registers, register_dim]
        """
        # Reshape registers to [total_registers, register_dim]
        registers = rearrange(self.registers, 'c n d -> (c n) d')
        
        # Repeat for batch
        registers = repeat(registers, 'n d -> b n d', b=batch_size)
        
        # Add spatial encoding if enabled
        if self.use_spatial_encoding:
            spatial_info = self.spatial_encoder(batch_size, device)
            # spatial_info shape: [batch_size, spatial_size^2, register_dim]
            # registers shape: [batch_size, total_registers, register_dim]
            # We need to make them compatible - take mean spatial info for each register
            spatial_summary = spatial_info.mean(dim=1, keepdim=True)  # [batch_size, 1, register_dim]
            spatial_summary = spatial_summary.expand(-1, registers.shape[1], -1)  # [batch_size, total_registers, register_dim]
            registers = registers + spatial_summary
        
        # Apply layer norm
        registers = self.norm(registers)
        
        return registers.to(device)
    
    def cross_attention_context(self, x, registers):
        """
        Prepare cross-attention context from registers.
        
        Args:
            x: Input features [b, c, h, w]
            registers: Anatomical registers [b, n, d]
            
        Returns:
            context: Cross-attention context [b, n, d]
        """
        b, c, h, w = x.shape
        
        # Apply projections
        q = self.to_q(registers)
        k = self.to_k(registers)
        v = self.to_v(registers)
        
        # Apply dropout
        context = self.dropout(v)
        
        return context
    
    def predict_anatomical_map(self, features):
        """
        Predict anatomical segmentation from features (for training supervision).
        
        Args:
            features: UNet features [b, c, h, w]
            
        Returns:
            pred_masks: Predicted segmentation logits [b, num_classes, h, w]
        """
        return self.anatomical_predictor(features)
    
    def get_anatomical_loss(self, pred_masks, target_masks):
        """
        Compute anatomical supervision loss.
        
        Args:
            pred_masks: Predicted masks [b, num_classes, h, w]
            target_masks: Ground truth masks [b, h, w] (class indices)
            
        Returns:
            loss: Cross-entropy loss
        """
        if target_masks is None:
            return torch.tensor(0.0, device=pred_masks.device)
        
        # Ensure target_masks is the right shape
        if target_masks.dim() == 4 and target_masks.shape[1] == 1:
            target_masks = target_masks.squeeze(1)
        
        # Resize predictions to match target size if needed
        if pred_masks.shape[2:] != target_masks.shape[1:]:
            pred_masks = F.interpolate(
                pred_masks, 
                size=target_masks.shape[1:], 
                mode='bilinear', 
                align_corners=False
            )
        
        return F.cross_entropy(pred_masks, target_masks.long())


class SpatialPositionEncoding(nn.Module):
    """
    Spatial position encoding for anatomical registers.
    Helps registers learn spatial relationships.
    """
    
    def __init__(self, dim, spatial_size):
        super().__init__()
        self.dim = dim
        self.spatial_size = spatial_size
        
        # Create learnable spatial embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, spatial_size * spatial_size, dim) * 0.02)
    
    def forward(self, batch_size, device):
        """
        Get spatial position encodings.
        
        Args:
            batch_size: Number of samples
            device: Device to place encodings on
            
        Returns:
            pos_encoding: [batch_size, spatial_size^2, dim]
        """
        pos_embed = repeat(self.pos_embed, '1 n d -> b n d', b=batch_size)
        return pos_embed.to(device)


class TimestepAwareRegisterBank(AnatomicalRegisterBank):
    """
    Extended register bank that modulates registers based on diffusion timestep.
    Allows different anatomical guidance at different stages of generation.
    """
    
    def __init__(self, *args, num_timesteps=1000, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_timesteps = num_timesteps
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.register_dim),
            nn.SiLU(),
            nn.Linear(self.register_dim, self.register_dim),
        )
        
        # Timestep-aware modulation
        self.time_modulation = nn.Sequential(
            nn.Linear(self.register_dim, self.register_dim * 2),
            nn.SiLU(),
        )
    
    def get_registers(self, batch_size, device, timestep=None):
        """
        Get timestep-modulated anatomical registers.
        """
        # Get base registers
        registers = super().get_registers(batch_size, device, timestep=None)
        
        if timestep is not None:
            # Embed timestep
            t_embed = self.time_embed(timestep.float().view(-1, 1))
            t_embed = repeat(t_embed, 'b d -> b n d', n=self.total_registers)
            
            # Modulate registers with timestep
            mod = self.time_modulation(t_embed)
            scale, shift = mod.chunk(2, dim=-1)
            registers = registers * (1 + scale) + shift
        
        return registers