import math
import torch
import torch.nn as nn
import torch.nn.functional as F


vocab_size = 56000
emb_dim = 1024
embd_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
attn_drop = nn.Dropout(0.1)
out_drop = nn.Dropout(0.1)


batch_size = 64
seq_len = 1024
inputs_inds = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

x = embd_layer(inputs_inds) # (bz, seq_len, emb_dim) = (64, 1024, 1024)

q_proj = nn.Linear(in_features=emb_dim, out_features=emb_dim) # Weight: (1024, 1024)
k_proj = nn.Linear(in_features=emb_dim, out_features=emb_dim) # Weight: (1024, 1024)
v_proj = nn.Linear(in_features=emb_dim, out_features=emb_dim) # Weight: (1024, 1024)
o_proj = nn.Linear(in_features=emb_dim, out_features=emb_dim) # Weight: (1024, 1024)

q = q_proj(x) # (bz, seq_len, emb_dim) = (64, 1024, 1024)
k = k_proj(x) # (bz, seq_len, emb_dim) = (64, 1024, 1024)
v = v_proj(x) # (bz, seq_len, emb_dim) = (64, 1024, 1024)

n_head = 16
d_k = emb_dim // n_head

q = q.view(batch_size, seq_len, n_head, d_k) # (bz, seq_len, n_head, d_k) = (64, 1024, 16, 64)
k = k.view(batch_size, seq_len, n_head, d_k) # (bz, seq_len, n_head, d_k) = (64, 1024, 16, 64) 
v = v.view(batch_size, seq_len, n_head, d_k) # (bz, seq_len, n_head, d_k) = (64, 1024, 16, 64)

q = q.transpose(1, 2) # (bz, n_head, seq_len, d_k) = (64, 16, 1024, 64)
k = k.transpose(1, 2) # (bz, n_head, seq_len, d_k) = (64, 16, 1024, 64)
v = v.transpose(1, 2) # (bz, n_head, seq_len, d_k) = (64, 16, 1024, 64)

# (bz, n_head, seq_len, d_k) x (bz, n_head, d_k, seq_len) => (bz, n_head, seq_len, seq_len)
attn = q @ k.transpose(2, 3) # (64, 16, 1024, 1024)

attn *= (1.0 / math.sqrt(k.size(-1)))
attn = F.softmax(attn, dim=-1)
attn = attn_drop(attn)

# (bz, n_head, seq_len, seq_len) x (bz, seq_len, emb_dim) => (bz, n_head, seq_len, emb_dim)
y = attn @ v # (64, 16, 1024, 64)

y = y.view(batch_size, seq_len, emb_dim) # (bz, seq_len, emb_dim) = (64, 1024, 1024)
y = o_proj(y) # (bz, seq_len, emb_dim) = (64, 1024, 1024)
y = out_drop(y)

