"""
Anatomical Latent Diffusion Model.
Extends the CompVis LatentDiffusion to integrate anatomical register learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from contextlib import contextmanager

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config


class AnatomicalLatentDiffusion(LatentDiffusion):
    """
    Anatomical-aware Latent Diffusion Model.
    
    Key features:
    1. Learns anatomical knowledge from segmentation masks during training
    2. Generates improved images without requiring masks at inference
    3. Integrates anatomical supervision with standard diffusion training
    4. Fully backward compatible with existing CompVis configs
    """
    
    def __init__(
        self,
        # Standard LatentDiffusion parameters
        first_stage_config,
        cond_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=False,  # Disable concat mode for anatomical registers
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        scale_by_std=False,
        
        # Anatomical-specific parameters
        anatomical_mask_key="segmentation",
        anatomical_loss_weight=1.0,
        progressive_anatomical_training=True,
        anatomical_warmup_steps=10000,
        use_anatomical_inference=True,
        
        *args, **kwargs
    ):
        # Force unconditional setup for anatomical registers
        if conditioning_key is None:
            conditioning_key = None  # Truly unconditional
        
        # Initialize parent class
        super().__init__(
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            num_timesteps_cond=num_timesteps_cond,
            cond_stage_key=cond_stage_key,
            cond_stage_trainable=cond_stage_trainable,
            concat_mode=concat_mode,
            cond_stage_forward=cond_stage_forward,
            conditioning_key=conditioning_key,
            scale_factor=scale_factor,
            scale_by_std=scale_by_std,
            *args, **kwargs
        )
        
        # Anatomical configuration
        self.anatomical_mask_key = anatomical_mask_key
        self.anatomical_loss_weight = anatomical_loss_weight
        self.progressive_anatomical_training = progressive_anatomical_training
        self.anatomical_warmup_steps = anatomical_warmup_steps
        self.use_anatomical_inference = use_anatomical_inference
        
        # Training step counter for progressive training
        self.register_buffer("anatomical_training_step", torch.tensor(0))
        
        # Check if model has anatomical registers
        from ldm.modules.diffusionmodules.anatomical_openaimodel import AnatomicalUNetModel
        
        self.has_anatomical_registers = isinstance(self.model.diffusion_model, AnatomicalUNetModel) and \
                                       hasattr(self.model.diffusion_model, 'use_anatomical_registers') and \
                                       self.model.diffusion_model.use_anatomical_registers
        
        
        if not self.has_anatomical_registers:
            print("Warning: UNet does not have anatomical registers enabled. Anatomical features will be disabled.")
    
    def forward(self, x, c, *args, **kwargs):
        """
        Forward pass with anatomical supervision.
        """
        # Get current global step for progressive training
        if hasattr(self, 'global_step'):
            # Update the buffer with current step
            with torch.no_grad():
                self.anatomical_training_step.copy_(torch.tensor(self.global_step, device=self.device))
        
        # Standard diffusion forward
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        
        return self.p_losses(x, c, t, *args, **kwargs)
    
    def p_losses(self, x_start, cond, t, noise=None, target_masks=None):
        """
        Compute losses including anatomical supervision.
        """
        noise = noise if noise is not None else torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Apply model with anatomical context
        result = self.apply_model(x_noisy, t, cond, target_masks=target_masks)
        if isinstance(result, tuple) and len(result) == 2:
            model_output, anatomical_loss = result
        else:
            model_output = result
            anatomical_loss = torch.tensor(0.0, device=x_start.device)
        
        # Defensive check - ensure model_output is a tensor
        while isinstance(model_output, (tuple, list)) and len(model_output) > 0:
            model_output = model_output[0]
        
        if not isinstance(model_output, torch.Tensor):
            raise ValueError(f"model_output should be a tensor, got {type(model_output)}")
        
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        
        # Standard diffusion loss
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()
        
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        
        # Apply learned variance weighting
        self.logvar = self.logvar.to(self.device)
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})
        
        loss = self.l_simple_weight * loss.mean()
        
        # VLB loss
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        
        # Add anatomical loss with progressive weighting
        if self.has_anatomical_registers and anatomical_loss > 0:
            anatomical_weight = self.get_anatomical_loss_weight()
            weighted_anatomical_loss = anatomical_loss * anatomical_weight
            loss += weighted_anatomical_loss
            
            loss_dict.update({
                f'{prefix}/anatomical_loss': anatomical_loss,
                f'{prefix}/anatomical_weight': anatomical_weight,
                f'{prefix}/weighted_anatomical_loss': weighted_anatomical_loss
            })
        
        loss_dict.update({f'{prefix}/loss': loss})
        
        return loss, loss_dict
    
    def get_anatomical_loss_weight(self):
        """
        Get anatomical loss weight with progressive training schedule.
        """
        if not self.progressive_anatomical_training:
            return self.anatomical_loss_weight
        
        # Progressive schedule: start high, decay to maintain balance
        step = self.anatomical_training_step.item()
        if step < self.anatomical_warmup_steps:
            # Start with strong anatomical supervision
            return self.anatomical_loss_weight * 2.0
        elif step < self.anatomical_warmup_steps * 2:
            # Gradually reduce to normal weight
            progress = (step - self.anatomical_warmup_steps) / self.anatomical_warmup_steps
            return self.anatomical_loss_weight * (2.0 - progress)
        else:
            # Maintain normal weight
            return self.anatomical_loss_weight
    
    def apply_model(self, x_noisy, t, cond, return_ids=False, target_masks=None):
        """
        Apply the UNet model with anatomical context.
        """
        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
        
        if self.has_anatomical_registers:
            # Call anatomical UNet directly, bypassing LatentDiffusion wrapper
            # Prepare context for the UNet
            context = None
            if cond:
                if 'c_crossattn' in cond:
                    context = cond['c_crossattn'][0] if isinstance(cond['c_crossattn'], list) else cond['c_crossattn']
                elif 'c_concat' in cond:
                    # For concat conditioning, we'd need to handle differently
                    context = cond['c_concat'][0] if isinstance(cond['c_concat'], list) else cond['c_concat']
            
            # Call UNet directly with target_masks
            result = self.model.diffusion_model(x_noisy, timesteps=t, context=context, target_masks=target_masks)
            
            if isinstance(result, tuple) and len(result) == 2:
                output, anatomical_loss = result
                if return_ids:
                    return output, anatomical_loss
                return output, anatomical_loss
            else:
                output = result
                anatomical_loss = torch.tensor(0.0, device=x_noisy.device)
                return output, anatomical_loss
        else:
            # Fall back to standard UNet
            output = self.model(x_noisy, t, **cond)
            anatomical_loss = torch.tensor(0.0, device=x_noisy.device)
            return output, anatomical_loss
    
    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, return_anatomical_masks=False):
        """
        Enhanced get_input that handles anatomical masks.
        """
        # Standard input processing
        x = batch[k]
        
            
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        
        # Ensure correct shape for medical images (add channel dim if needed)
        if x.dim() == 3:  # [B, H, W] -> [B, 1, H, W]
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[1] > 1:
            # If we have multiple channels but encoder expects 1, take first channel
            if hasattr(self.first_stage_model.encoder, 'conv_in') and self.first_stage_model.encoder.conv_in.in_channels == 1:
                x = x[:, 0:1, :, :]
        
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        
        # Handle standard conditioning
        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox']:
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    xc = batch[cond_key]
                    if isinstance(xc, torch.Tensor):
                        xc = xc.to(self.device)
            else:
                xc = x
            
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]
        else:
            c = None
            xc = None
        
        # Handle anatomical masks
        anatomical_masks = None
        if self.anatomical_mask_key in batch and self.has_anatomical_registers:
            if isinstance(batch[self.anatomical_mask_key], torch.Tensor):
                anatomical_masks = batch[self.anatomical_mask_key]
            else:
                # Handle numpy arrays or other formats
                anatomical_masks = torch.tensor(batch[self.anatomical_mask_key])
            
            anatomical_masks = anatomical_masks.to(self.device)
            if bs is not None:
                anatomical_masks = anatomical_masks[:bs]
            
            # Ensure masks are in the right format [B, H, W] with class indices
            if anatomical_masks.dim() == 4 and anatomical_masks.shape[1] == 1:
                anatomical_masks = anatomical_masks.squeeze(1)
            elif anatomical_masks.dim() == 4:
                # Convert from one-hot to class indices if needed
                anatomical_masks = anatomical_masks.argmax(dim=1)
        
        # Build output
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        if return_anatomical_masks:
            out.append(anatomical_masks)
        
        return out
    
    def shared_step(self, batch, **kwargs):
        """
        Training step with anatomical supervision.
        """
        if self.has_anatomical_registers:
            # Get input with anatomical masks
            x, c, *rest = self.get_input(batch, self.first_stage_key, return_anatomical_masks=True)
            anatomical_masks = rest[-1] if rest else None
            
            # Forward pass with anatomical supervision - returns (loss, loss_dict)
            loss, loss_dict = self(x, c, target_masks=anatomical_masks)
            return loss, loss_dict
        else:
            # Standard training without anatomical features
            x, c = self.get_input(batch, self.first_stage_key)
            result = self(x, c)
            if isinstance(result, tuple) and len(result) == 2:
                loss, loss_dict = result
                return loss, loss_dict
            else:
                # If forward only returns loss, create minimal loss_dict
                loss = result
                prefix = 'train' if self.training else 'val'
                loss_dict = {f'{prefix}/loss': loss}
                return loss, loss_dict
    
    @torch.no_grad()
    def sample(self, cond=None, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None, **kwargs):
        """
        Sampling with anatomical register guidance (no masks required).
        """
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        
        # For anatomical LDM, we don't need conditioning at inference
        # The anatomical registers provide implicit structural guidance
        if self.has_anatomical_registers and self.use_anatomical_inference:
            cond = None  # Rely on anatomical registers
        
        return self.p_sample_loop(
            cond,
            shape,
            return_intermediates=return_intermediates,
            x_T=x_T,
            verbose=verbose,
            timesteps=timesteps,
            quantize_denoised=quantize_denoised,
            mask=mask,
            x0=x0
        )
    
    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1.,
                   return_keys=None, quantize_denoised=True, inpaint=True, plot_denoise_rows=False,
                   plot_progressive_rows=True, plot_diffusion_rows=True, **kwargs):
        """
        Enhanced logging that includes anatomical visualizations.
        """
        use_ddim = ddim_steps is not None
        
        log = dict()
        
        # Get inputs with anatomical masks if available
        if self.has_anatomical_registers:
            z, c, x, xrec, xc, anatomical_masks = self.get_input(
                batch, self.first_stage_key,
                return_first_stage_outputs=True,
                force_c_encode=True,
                return_original_cond=True,
                bs=N,
                return_anatomical_masks=True
            )
        else:
            z, c, x, xrec, xc = self.get_input(
                batch, self.first_stage_key,
                return_first_stage_outputs=True,
                force_c_encode=True,
                return_original_cond=True,
                bs=N
            )
            anatomical_masks = None
        
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        
        # Log anatomical masks if available
        if anatomical_masks is not None:
            # Convert class indices to RGB for visualization
            anatomical_rgb = self.masks_to_rgb(anatomical_masks)
            log["anatomical_masks"] = anatomical_rgb
        
        # Standard conditioning visualization
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                from ldm.util import log_txt_as_img
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                from ldm.util import log_txt_as_img
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif self.cond_stage_key == 'segmentation':
                # For segmentation conditioning, visualize the processed conditioning
                log["conditioning"] = xc
        
        # Sampling
        if sample:
            # Sample with anatomical register guidance
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(
                    cond=None,  # Unconditional sampling with anatomical registers
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps
                )
            
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            
            if plot_denoise_rows:
                denoise_row = self._get_denoise_row_from_list(z_denoise_row, desc="Denoise Row")
                log["denoise_row"] = denoise_row
        
        # Additional diffusion visualization
        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t_tensor = torch.full((n_row,), t, device=self.device, dtype=torch.long)
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t_tensor, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))
            
            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            from einops import rearrange
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            from torchvision.utils import make_grid
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid
        
        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        
        return log
    
    def masks_to_rgb(self, masks):
        """
        Convert class index masks to RGB for visualization.
        """
        # Simple colormap for visualization
        colors = torch.tensor([
            [0, 0, 0],      # background - black
            [255, 0, 0],    # class 1 - red
            [0, 255, 0],    # class 2 - green
            [0, 0, 255],    # class 3 - blue
            [255, 255, 0],  # class 4 - yellow
            [255, 0, 255],  # class 5 - magenta
            [0, 255, 255],  # class 6 - cyan
            [128, 128, 128], # class 7 - gray
        ], device=masks.device, dtype=torch.float32)
        
        # Extend colors if needed
        max_class = masks.max().item()
        if max_class >= len(colors):
            # Generate random colors for additional classes
            additional_colors = torch.randint(0, 256, (max_class + 1 - len(colors), 3), 
                                            device=masks.device, dtype=torch.float32)
            colors = torch.cat([colors, additional_colors], dim=0)
        
        # Map masks to RGB
        rgb_masks = colors[masks.long()]  # [B, H, W, 3]
        rgb_masks = rgb_masks.permute(0, 3, 1, 2) / 255.0  # [B, 3, H, W], normalize to [0,1]
        
        return rgb_masks