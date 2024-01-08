## QLoRA: Efficient Finetuning of Quantized LLMs
- Link paper: https://arxiv.org/abs/2305.14314
- Github: https://github.com/artidoro/qlora

### Keywords:
- 4-bit NormalFloat (NF4)
- Double Quantization
- Paged Optimizers - avoid the gradient checkpointing "memory spikes"
- Vicuna benchmark. https://lmsys.org/blog/2023-03-30-vicuna

### Background
#### 1. Block-wise k-bit Quantization
- Flattening the input tensor $X \in R^{b \times h}$
- Slicing the linear segment into $n = (b \times h) / B$ block
- Quantize these blocks independently
- Each block has a quantization constant $c_{i}$ independent (dtype = FP32)

#### 2. Double Quantization
- Quantize for constant $c_{i}$ to save bits.

#### 3. k-bit NormalFloat (NF4)
- Over 90% pre-trained weight have center-zero distribution.
- Weighted the weight after quantize by a Normal Distribution.

#### 4. Low-rank Adapters
- LoRA with r higher (r=16)

#### 5. Paged Optimizers
- Bring some memory not using at a time such as (Optimizer states) to CPU memory (RAM). When the optimizer update gradient, the states will bring back to GPU.
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/

### QLORA Finetuning
- 4-bit for storage
- bfloat16 for computation
