## QLoRA: Efficient Finetuning of Quantized LLMs
- Link paper: https://arxiv.org/abs/2305.14314

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

#### 2. Low-rank Adapters
- LoRA 

#### 3. Memory Requirement of Parameter-Efficient Finetuning
- Gradient checkpointing. https://arxiv.org/abs/1604.06174
- Aggressively reducing the amount of LoRA parameter yields only minor memory benefits.


### QLORA Finetuning
- 4-bit for storage
- bfloat16 for computation

#### 1. 4-bit NormalFloat Quantization
 - 8-bit Optimizers via Block-wise Quantization. https://arxiv.org/abs/2110.02861v1


#### 2. Double Quantization


#### 2. Paged Optimizers
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/


