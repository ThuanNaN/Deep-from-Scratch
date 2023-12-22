## LLaMA: Open and Efficient Foundation Language Models
- Link paper: https://arxiv.org/abs/2302.13971

### Keywords:

### Tokenizer
- byte-pair encoding (BPE) algorithm.

### Architecture
Based on the transformer architecture with some changes:
- Pre-normalization [GPT3]
    - RMSNorm. https://arxiv.org/abs/1910.07467
- SwiGLU activation function [PaLM]
    - https://arxiv.org/abs/2002.05202
- Rotary Embeddings [GPTNeo]
    - https://arxiv.org/abs/2104.09864

### Optimizer
- AdamW ($\beta_1=0.9$, $\beta_2=0.95$, weight_decay = 0.1)

- Cosine learning rate schedule (final = 10% maximum lr)
- Cliping = 0.1
- Lr = {1.5e-4, 3e-4}

### Efficient implementation
- https://github.com/facebookresearch/xformers


### Massive Multitask Language Understanding
- Five-shot accuracy

### Instruction Finetuning
- Scaling Instruction-Finetuned Language Models. https://arxiv.org/abs/2210.11416