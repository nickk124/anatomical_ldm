"""
Anatomical-aware UNet model that extends the CompVis OpenAI UNet.
Integrates anatomical register cross-attention for structure-aware generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ldm.modules.diffusionmodules.openaimodel import UNetModel, AttentionBlock, ResBlock, TimestepEmbedSequential
from ldm.modules.attention import SpatialTransformer
from ldm.modules.anatomical_registers import AnatomicalRegisterBank, TimestepAwareRegisterBank


class AnatomicalUNetModel(UNetModel):
    """
    Extended UNet with anatomical register cross-attention.
    
    Key differences from base UNet:
    1. Adds anatomical register bank
    2. Integrates anatomical cross-attention in attention layers
    3. Adds anatomical prediction head for training supervision
    4. Maintains full backward compatibility with existing configs
    """
    
    def __init__(
        self,
        # Standard UNet parameters
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        n_embed=None,
        legacy=True,
        
        # Anatomical register parameters
        use_anatomical_registers=True,
        num_anatomical_classes=None,
        anatomical_register_dim=None,
        num_registers_per_class=4,
        anatomical_loss_weight=1.0,
        use_timestep_aware_registers=False,
        anatomical_attention_heads=8,
        **kwargs
    ):
        # Initialize base UNet
        super().__init__(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            num_classes=num_classes,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer,
            transformer_depth=transformer_depth,
            context_dim=context_dim,
            n_embed=n_embed,
            legacy=legacy,
        )
        
        # Anatomical register configuration
        self.use_anatomical_registers = use_anatomical_registers
        self.anatomical_loss_weight = anatomical_loss_weight
        
        if use_anatomical_registers:
            assert num_anatomical_classes is not None, "Must specify num_anatomical_classes when using anatomical registers"
            
            # Default register dimension to model channels if not specified
            if anatomical_register_dim is None:
                anatomical_register_dim = model_channels
            
            # Create anatomical register bank
            register_class = TimestepAwareRegisterBank if use_timestep_aware_registers else AnatomicalRegisterBank
            self.anatomical_registers = register_class(
                num_classes=num_anatomical_classes,
                register_dim=anatomical_register_dim,
                num_registers_per_class=num_registers_per_class,
                spatial_dim=image_size,  # Latent spatial dimension
            )
            
            # Add anatomical cross-attention to existing attention layers
            self._add_anatomical_attention(anatomical_register_dim, anatomical_attention_heads)
            
            # Anatomical prediction head for training supervision
            # Use features from the middle block for prediction
            self.anatomical_predictor = nn.Sequential(
                nn.GroupNorm(32, model_channels * channel_mult[-1]),
                nn.SiLU(),
                nn.Conv2d(model_channels * channel_mult[-1], model_channels, 3, padding=1),
                nn.GroupNorm(32, model_channels),
                nn.SiLU(),
                nn.Conv2d(model_channels, num_anatomical_classes, 1),
            )
    
    def _add_anatomical_attention(self, register_dim, num_heads):
        """
        Add anatomical cross-attention to existing attention blocks.
        """
        self.anatomical_cross_attn_blocks = nn.ModuleList()
        
        # Find all attention blocks in the UNet and add anatomical cross-attention
        for module in self.modules():
            if isinstance(module, (AttentionBlock, SpatialTransformer)):
                # Get the number of channels for this attention block
                if isinstance(module, AttentionBlock):
                    channels = module.channels
                else:  # SpatialTransformer
                    channels = module.in_channels
                
                # Create anatomical cross-attention for this block
                anatomical_attn = AnatomicalCrossAttention(
                    channels=channels,
                    register_dim=register_dim,
                    num_heads=num_heads,
                )
                self.anatomical_cross_attn_blocks.append(anatomical_attn)
    
    def forward(self, x, timesteps=None, context=None, y=None, target_masks=None, **kwargs):
        """
        Forward pass with anatomical register integration.
        
        Args:
            x: Input tensor [b, c, h, w]
            timesteps: Diffusion timesteps [b]
            context: Standard cross-attention context (from existing conditioning)
            y: Class labels (if class-conditional)
            target_masks: Ground truth masks for training [b, h, w] (optional)
            
        Returns:
            output: Denoised output [b, c, h, w]
            anatomical_loss: Anatomical supervision loss (if target_masks provided)
        """
        anatomical_loss = torch.tensor(0.0, device=x.device)
        
        # Get anatomical registers if enabled
        anatomical_context = None
        if self.use_anatomical_registers:
            anatomical_registers = self.anatomical_registers.get_registers(
                batch_size=x.shape[0],
                device=x.device,
                timestep=timesteps
            )
            anatomical_context = self.anatomical_registers.cross_attention_context(x, anatomical_registers)
        
        # Standard timestep embedding
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        
        hs = []
        t_emb = self._get_timestep_embedding(timesteps, x.device)
        emb = self.time_embed(t_emb)
        
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        
        # Forward through input blocks with anatomical attention
        h = x.type(self.dtype)
        anatomical_attn_idx = 0
        
        for module in self.input_blocks:
            h = module(h, emb, context)
            
            # Apply anatomical cross-attention if this block has attention
            if self.use_anatomical_registers and self._has_attention(module):
                if anatomical_attn_idx < len(self.anatomical_cross_attn_blocks):
                    h = self.anatomical_cross_attn_blocks[anatomical_attn_idx](h, anatomical_context)
                    anatomical_attn_idx += 1
            
            hs.append(h)
        
        # Middle block with anatomical attention
        h = self.middle_block(h, emb, context)
        if self.use_anatomical_registers and anatomical_attn_idx < len(self.anatomical_cross_attn_blocks):
            h = self.anatomical_cross_attn_blocks[anatomical_attn_idx](h, anatomical_context)
            anatomical_attn_idx += 1
        
        # Capture middle block features for anatomical prediction
        middle_features = h
        
        # Forward through output blocks with anatomical attention
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
            
            # Apply anatomical cross-attention if this block has attention
            if self.use_anatomical_registers and self._has_attention(module):
                if anatomical_attn_idx < len(self.anatomical_cross_attn_blocks):
                    h = self.anatomical_cross_attn_blocks[anatomical_attn_idx](h, anatomical_context)
                    anatomical_attn_idx += 1
        
        h = h.type(x.dtype)
        
        # Generate final output
        if self.predict_codebook_ids:
            output = self.id_predictor(h)
        else:
            output = self.out(h)
        
        # Compute anatomical loss if target masks provided
        if self.use_anatomical_registers and target_masks is not None and self.training:
            pred_masks = self.anatomical_predictor(middle_features)
            anatomical_loss = self.anatomical_registers.get_anatomical_loss(pred_masks, target_masks)
            anatomical_loss = anatomical_loss * self.anatomical_loss_weight
        
        return output, anatomical_loss
    
    def _get_timestep_embedding(self, timesteps, device):
        """Get timestep embedding (copied from parent class for compatibility)."""
        from ldm.modules.diffusionmodules.util import timestep_embedding
        return timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    
    def _has_attention(self, module):
        """Check if a module contains attention blocks."""
        for submodule in module.modules():
            if isinstance(submodule, (AttentionBlock, SpatialTransformer)):
                return True
        return False


class AnatomicalCrossAttention(nn.Module):
    """
    Cross-attention module for integrating anatomical register context.
    """
    
    def __init__(self, channels, register_dim, num_heads=8):
        super().__init__()
        self.channels = channels
        self.register_dim = register_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"
        
        # Project input features to query
        self.to_q = nn.Linear(channels, channels)
        
        # Project anatomical registers to key and value
        self.to_k = nn.Linear(register_dim, channels)
        self.to_v = nn.Linear(register_dim, channels)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Dropout(0.1)
        )
        
        # Normalization
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(register_dim)
    
    def forward(self, x, anatomical_context):
        """
        Apply anatomical cross-attention.
        
        Args:
            x: Input features [b, c, h, w]
            anatomical_context: Anatomical register context [b, n_registers, register_dim]
            
        Returns:
            output: Features with anatomical attention applied [b, c, h, w]
        """
        b, c, h, w = x.shape
        
        # Reshape input to sequence format
        x_seq = x.reshape(b, c, h * w).transpose(1, 2)  # [b, h*w, c]
        
        # Normalize inputs
        q = self.norm_q(x_seq)
        kv = self.norm_kv(anatomical_context)
        
        # Project to q, k, v
        q = self.to_q(q)  # [b, h*w, c]
        k = self.to_k(kv)  # [b, n_registers, c]
        v = self.to_v(kv)  # [b, n_registers, c]
        
        # Reshape for multi-head attention
        q = q.reshape(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)  # [b, heads, h*w, head_dim]
        k = k.reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)     # [b, heads, n_registers, head_dim]
        v = v.reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)     # [b, heads, n_registers, head_dim]
        
        # Compute attention scores
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [b, heads, h*w, n_registers]
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)  # [b, heads, h*w, head_dim]
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).reshape(b, h * w, c)  # [b, h*w, c]
        
        # Output projection
        output = self.to_out(attn_output)
        
        # Residual connection
        output = output + x_seq
        
        # Reshape back to spatial format
        output = output.transpose(1, 2).reshape(b, c, h, w)
        
        return output