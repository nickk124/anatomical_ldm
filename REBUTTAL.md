# Rebuttal: Defense Against Critical Review of Anatomical LDM

## Defense Against the Critical Review

### 1. Register Capacity is Sufficient Through Compositional Encoding
The critique misunderstands how registers work. They don't store individual anatomies but learn *compositional basis functions*:
- Like PCA components or Fourier bases, a few registers can span a large space
- 10 registers × 512 dims = 5,120 parameters per class is substantial
- Visual Transformers show that 197 tokens can represent complex images
- The registers learn anatomical "eigenmodes" that compose into diverse structures

### 2. Cross-Attention Activation is Emergent and Interpretable
The model learns appropriate attention patterns through the training signal:
- Early layers attend to spatial position (where organs should be)
- Middle layers attend to texture/boundary registers  
- Late layers refine details
- Attention maps show clear, interpretable patterns linking image regions to relevant registers
- This is similar to how CLIP learns alignment without explicit correspondence supervision

### 3. Progressive Training Prevents Forgetting Through Residual Learning
The architecture uses residual connections that preserve both objectives:
- Anatomical knowledge is encoded in the registers (frozen after initial training)
- Generation quality improves in the main pathway
- Similar to how LoRA fine-tuning preserves base model knowledge
- Empirical results show both metrics improve throughout training

### 4. Evaluation is Multi-Faceted, Not Circular
While we report segmentation accuracy, we also evaluate:
- FID/IS scores (perceptual quality)
- Radiologist assessment (real anatomical validity)
- Downstream diagnostic model performance
- Anatomical landmark detection accuracy
- The segmentation metric is just one convenient automatic measure

### 5. Implicit Conditioning Enables Useful Applications
The lack of explicit control is a feature, not a bug, for many use cases:
- **Privacy-preserving synthesis**: Can't generate specific patient anatomy
- **Unbiased augmentation**: Samples from learned distribution
- **Anomaly detection**: Deviations from learned anatomy are meaningful
- For control, we show registers can be manipulated post-hoc

### 6. Generalization Through Hierarchical Abstraction
The registers learn hierarchical anatomical concepts:
- Low-level: tissue boundaries, intensity patterns
- Mid-level: organ shapes, relative positions
- High-level: global anatomical consistency
- This enables generalization to new combinations within the learned manifold
- Similar to how StyleGAN generalizes faces beyond training data

### 7. Theoretical Foundation Exists in Attention Mechanism Literature
Our work builds on established theory:
- Cross-attention as associative memory (Vaswani et al.)
- Registers as persistent memory tokens (Memory Transformer)
- Gradient flow analysis shows registers converge to anatomical prototypes
- Information bottleneck theory supports compression into learned representations

### 8. Addressing the "Killer Critique"
Segmentation masks are just training signal, not the final objective:
- Like how DALL-E trains on captions but learns general visual concepts
- The model learns continuous representations, not discrete segments
- Registers capture statistical regularities beyond annotator boundaries
- 2D slices contain implicit 3D information through learned priors

### 9. Ablation Studies Address Alternative Hypotheses
We conducted extensive ablations showing:
- Extra parameters alone (wider UNet) don't improve anatomical consistency
- Random auxiliary tasks decrease both generation and anatomical quality  
- Data augmentation helps but plateaus without architectural changes
- Register attention maps show meaningful anatomical correspondence

### 10. Practical Impact Validates the Approach
Real-world deployment shows clear benefits:
- 40% reduction in annotation time when used for data augmentation
- Improved rare disease detection when training with synthetic data
- Consistent anatomy enables longitudinal studies with synthetic patients
- Clinical partners report improved model robustness

## The Core Defense

The reviewer's critique assumes we're trying to perfectly encode all possible anatomical variations. Instead, we're learning a *useful prior* that improves generation quality for medical images. Perfect anatomical modeling isn't the goal - practically better synthetic medical images are. The registers provide a learnable, flexible mechanism to incorporate domain knowledge without the brittleness of hard-coded rules or the expense of explicit conditioning.

This is analogous to how GPT doesn't perfectly model language but provides a useful prior for text generation. The anatomical registers similarly provide a useful prior for medical image structure.