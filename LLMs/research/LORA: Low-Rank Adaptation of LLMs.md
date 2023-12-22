## LoRA: Low-Rank Adaptation of Large Language Models
- Link paper: https://arxiv.org/abs/2106.09685
-Github: https://github.com/microsoft/LoRA

### Keywords:


### Method
- Pre-trained weight $W_0 \in R^{d \times k}$

- Update: $W_0 + \Delta W = W_0 + BA$ \
    Where: $B \in R^{d \times r}, A \in R^{r \times k}$, and $r << min(d, k)$

- During training, $W_0$ is frozen. Only A and B contain trainable parameters.
- $h = W_0 x \to h = W_0x + BAx$
- Initialization:
    - Gaussian for A
    - Zero for B
- Scaling:
    - scale $\Delta W$ by $\frac{\alpha}{r}$