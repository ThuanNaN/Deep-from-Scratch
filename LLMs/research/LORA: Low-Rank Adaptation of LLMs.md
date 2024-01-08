## LoRA: Low-Rank Adaptation of Large Language Models
- Link paper: https://arxiv.org/abs/2106.09685
- Github: https://github.com/microsoft/LoRA

### Keywords:
- Rank Matrix Decomposition
- Unlike adapters, no additional inference latency.
- Only adapting the attention weights

### Method
- Low-Rank Adaptation, or LoRA, which freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture.

- Learned weights can be merged with the main weights during inference, thus not introducing any latency

- Pre-trained weight $W_0 \in R^{d \times k}$

- Update: $W_0 + \Delta W = W_0 + BA$ \
    Where: $B \in R^{d \times r}, A \in R^{r \times k}$, and $r << min(d, k)$

- During training, $W_0$ is frozen. Only A and B contain trainable parameters.
- $h = W_0 x \to h = W_0x + BAx$
- Initialization:
    - Gaussian for A (nn.init.normal_)
    - Zero for B (nn.init.zeros_)
    - $\Delta$ W = BA is zero at the beginning of training
- Scaling:
    - scale $\Delta W$ by $\frac{\alpha}{r}$
  
### UNDERSTANDING THE LOW-RANK UPDATES
- WHICH WEIGHT MATRICES IN TRANSFORMER SHOULD WE APPLY LORA TO? 
  - $W_k, W_v$
- WHAT IS THE OPTIMAL RANK r FOR LORA?
- HOW DOES THE ADAPTATION MATRIX ∆W COMPARE TO W?
