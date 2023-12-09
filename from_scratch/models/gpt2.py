"""
OpenAI (Tensorflow): https://github.com/openai/gpt-2/blob/master/src/model.py
Huggingface (Pytorch): https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
NanoGPT (Pytorch): https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT has 50257 vocab
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight =nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, input):
        return F.layer_norm(input, self.weight.size(), self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # MultiHead
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias=config.bias)
        # Output Projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            print("Fash attention require Pytorch >= 2.0. Using slow attention")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                              .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch_size, seq_len, emb_dim
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, 
                                           attn_mask=None, 
                                           dropout_p=self.dropout if self.training else 0.0, 
                                           is_causal=True)
        else:
            attn = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
            attn = attn.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_dropout(self.c_proj(y))
        return y


class GPT2MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self,
                 config: GPTConfig
                 ) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = GPT2MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            emb_drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weight)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        pos =  torch.arange(0, t, dtype=torch.long, device=device)

        token_embd = self.transformer.wte(input_ids)
        pos_embd = self.transformer.wpe(pos)
        x = self.transformer.emb_drop(token_embd + pos_embd)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        loss = None
        if targets is not None:
            logits =self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
    


    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= self.config.block_size else input_ids[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits<v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((input_ids, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    config = GPTConfig()
    model = GPT2Model(config)

    batch_size = 16
    batch_data = torch.randint(low=0, high=config.vocab_size, 
                               size=(batch_size, config.block_size), 
                               dtype=torch.long)

    logits, loss = model(batch_data)
    print(logits.size())