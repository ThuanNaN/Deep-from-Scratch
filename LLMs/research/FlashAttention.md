# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- Paper: https://arxiv.org/abs/2205.14135
- Github: https://github.com/Dao-AILab/flash-attention

## Keywords
- GPU high bandwidth memory (HBM). https://en.wikipedia.org/wiki/High_Bandwidth_Memory
- sparse-approximation [51, 74]
- low-rank approximation [12, 50, 84]
- combinations [3, 9, 92]
- IO-aware [1]
- HBM [45]


## Concepts
### 1. 
### 2. 
### 2. 
### 4. 
### 5. 

## Backgrounds
### 1. Hardware Performance
#### 1.1 GPU Memory Hierarchy
- HBM larger (40 or 80GB on A100) - 1.5->2TB/s but lower than on-chip SRAM (192Kb) 15TB/s
#### 1.2 Execution Model
- GPUs have a massive number of threads to execute an operation (called a kernel).
Each kernel loads inputs from HBM to registers and SRAM, computes, then writes outputs to HBM.
#### 1.3 Performance characteristics
- arithmetic intensity: the number of arithmetic operations per byte of memory access.
- compute-bound: the time taken by the operation is determined by how many arithmetic operations there
are, while time accessing HBM is much smaller. (eg. Matrix matmul with high dims, Conv with high chanels)
- memory-bound: he time taken by the operation is determined by the number of memory accesses, while
time spent in computation is much smaller. (eg. softmax, sum, batch norm, layer norm)
#### 1.4 Kernel fusion.
- if there are multiple operations applied to the same input, the input can be loaded once from HBM, instead of multiple times for each operation
## 2. Standard Attention Implementation
- Require: Matrices Q, K, V in HBM
- Load Q, K by blocks from HBM, compute S = QK, write S to HBM
- Read S from HBM, compute P = softmax(S), write P to HBM
- Load P and V by blocks from HBM, compute O = PV, write O to HBM
- Return O
## 3. FlashAttention
### 3.1 
### 3.2 
### 3.3

### Ref
- [1]
- [3]
- [9]
- [12]
- [45] Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking - https://arxiv.org/pdf/1804.06826.pdf
- [50]
- [51]
- [74] 
- [84]
- [92]