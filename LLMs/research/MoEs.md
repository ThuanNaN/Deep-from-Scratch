# Mixture of Experts (MoE)

## 1. What is MoE
MoE consists of two main elements:
- Sparse MoE layers are used instead of dense feed-forward network (FFN) layers.
- A gate network or router, that determines which tokens are sent to which expert.


(token + pos_emb) -> Attention -> (Add & Norm) -> (MoEs) -> (Add & Norm)

(MoEs):
token -> (Gate / Router) -> Choice (one/more)[Expert_1, Expert_2, ... Expert_n]

Expert <=> FFN

Think: chunk the FFN (weight) into N part which each part is a expert. Training a large params (same with FNN) but at inference time, only some expert join to compute -> Faster than FNN. But all expert need load to VRAM.

## 2.

## 3.


## 4.
