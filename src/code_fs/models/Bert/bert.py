# Paper: https://arxiv.org/abs/1810.04805
# github: https://github.com/codertimo/BERT-pytorch

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class BertConfig:
    d_model: int = 512
    n_embd: int = 512
    max_len: int = 512
    vocab_size: int = 32000
    n_head: int = 8
    n_layer: int = 6
    forward_expansion: int = 4
    dropout: float = 0.1
    bias: bool = False
    

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size, pad_idx=0) -> None:
        super().__init__(vocab_size, embed_size, padding_idx=pad_idx)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe[:, :x.size(1)]


class SegementEmbedding(nn.Embedding):
    def __init__(self, embed_size, pad_idx=0) -> None:
        super().__init__(3, embed_size, padding_idx=pad_idx)


class BertEmbedding(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.wte = TokenEmbedding(config.vocab_size, config.n_embd)
        self.wpe = PositionalEmbedding(config.n_embd, max_len=512)
        self.wse = SegementEmbedding(config.n_embd)

        self.dropout = nn.Dropout(config.dropout)
        self.embed_size = config.n_embd

    def forward(self, sequence, seqment_label):
        x = self.wte(sequence) + self.wpe(sequence) + self.wse(seqment_label)
        return self.dropout(x)


class SelfAttention(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.d_k = config.d_model // config.n_head

        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)

        self.c_attn = nn.Linear(config.d_model, config.d_model*3, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.bias = config.bias

    def forward(self, x, mask=None):
        B, T, C = x.size() # batch_size, seq_len, emb_dim
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        attn = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
        if mask is not None:
            attn = attn.masked_fill(mask==0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y = attn @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_dropout(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        ffn_hidden = config.n_embd * config.forward_expansion

        self.feed_forward = nn.Sequential(
            nn.Linear(config.n_embd, ffn_hidden, bias=config.bias),
            nn.ReLU(),
            nn.Linear(ffn_hidden, config.n_embd, bias=config.bias)
        )

    def forward(self, x):
        return self.feed_forward(x)
        

class TransformerBlock(nn.Module):
    def __init__(self, config:BertConfig) -> None:
        super().__init__()
        self.attn = SelfAttention(config)
        self.ffn = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.norm2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return self.dropout(x)


class Bert(nn.Module):
    def __init__(self, config:BertConfig) -> None:
        super().__init__()

        self.embedding = BertEmbedding(config)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])


    def forward(self, input_ids, segment_info):
        mask = (input_ids > 0).unsqueeze(1).repeat(1, input_ids.size(1), 1).unsqueeze(1)
        x = self.embedding(input_ids, segment_info)
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        return x
    


if __name__ == "__main__":
    config = BertConfig()
    model = Bert(config)

    batch_size = 16
    input_ids = torch.randint(low=0, high=32000, size=(batch_size, config.max_len))
    segment_info = torch.randint(0, 2, (batch_size, config.max_len))
    output = model(input_ids, segment_info)
    print(output.shape)




