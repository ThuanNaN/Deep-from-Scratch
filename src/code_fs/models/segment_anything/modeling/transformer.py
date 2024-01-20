from typing import Type, Tuple
import torch
from torch import nn, Tensor
from .common import MLPBlock


class TwoWayTransformer(nn.Module): 
    """
    A transformer decoder that attends to an input image using
    queries whose positional embedding is supplied.
    """
    def __init__(self,
                 depth: int,
                 emded_dim: int,
                 num_heads: int,
                 mlp_dim: int,  
                 actiation: Type[nn.Module] = nn.ReLU,
                 attention_downsample_rate: int = 2, 
                 ) -> None:
        super().__init__()
        self.depth = depth
        self.embed_dim = emded_dim
        self.num_heads = num_heads  
        self.mlp_dim = mlp_dim  
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    emded_dim,
                    num_heads,
                    mlp_dim,
                    actiation,
                    attention_downsample_rate,
                    skip_first_layer_pe=(i == 0)
                )
            )
        self.final_attn_token2img = Attention(emded_dim, num_heads, attention_downsample_rate)
        self.final_norm = nn.LayerNorm(emded_dim)

    def forward(self,
                image_embed: Tensor, 
                image_pe: Tensor,
                point_embed: Tensor,
                ) -> Tuple[Tensor, Tensor]:

        bs, c, h, w = image_embed.shape
        # merege the H and W dim, then convert the channel dim to the last dim
        image_embed = image_embed.flatten(2).permute(0, 2, 1) 
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embed
        keys = image_embed

        for layer in self.layers:  
            queries, keys = layer(queries = queries, 
                                  keys = keys, 
                                  query_pe = point_embed, 
                                  key_pe = image_pe)
        
        q = queries + point_embed
        k = keys + image_pe
        attn_out = self.final_attn_token2img(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.final_norm(queries)

        return queries, keys

class TwoWayAttentionBlock(nn.Module):
    """
    A transformer block with four layers: (1) self-attention of sparse
    inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
    block on sparse inputs, and (4) cross attention of dense inputs to sparse
    inputs.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_dim: int,
                 actiation: Type[nn.Module] = nn.ReLU,
                 attention_downsample_rate: int = 2,
                 skip_first_layer_pe: bool = False, # skip the PE on the first layer
                 ) -> None:
        super().__init__()
        self.self_attn = Attention(embed_dim, num_heads, attention_downsample_rate)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attention_token2img = Attention(embed_dim, num_heads, attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = MLPBlock(embed_dim, mlp_dim, actiation)
        self.norm3 = nn.LayerNorm(embed_dim)    

        self.cross_attention_img2token = Attention(embed_dim, num_heads, attention_downsample_rate)
        self.norm4 = nn.LayerNorm(embed_dim)

        self.skip_first_layer_pe = skip_first_layer_pe
    
    def forward(self,
                queries: Tensor,
                keys: Tensor,
                query_pe: Tensor,  
                key_pe: Tensor
                ) -> Tuple[Tensor, Tensor]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # cross attention block, tokens attending image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attention_token2img(q=q, k=k, v=keys)
        queries = queries + attn_out    
        queries = self.norm2(queries)

        # mlp block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # cross attention block, image embedding attending tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attention_img2token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    Downscaling the size of the embedding
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 downsample_rate: int
                 ) -> None:
        super().__init__() 
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.internal_dim = embed_dim // downsample_rate
        assert self.internal_dim % num_heads == 0, "num_heads must devide embed_dim"

        self.q_proj = nn.Linear(embed_dim, self.internal_dim)
        self.k_proj = nn.Linear(embed_dim, self.internal_dim)
        self.v_proj = nn.Linear(embed_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embed_dim)
    
    def _seperate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2) # (b, n_heads, n_tokens, c_per_head)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head) #(b, n_tokens, embed_dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # input projection
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # seperate heads
        q = self._seperate_heads(q, self.num_heads)
        k = self._seperate_heads(k, self.num_heads)
        v = self._seperate_heads(v, self.num_heads) 

        # attention
        c_per_head = q.shape[-1]
        attn = q @ k.permute(0, 1, 3, 2) # (b, n_heads, c_per_head, n_tokens)
        attn = attn / (c_per_head ** 0.5)
        attn = torch.softmax(attn, dim=-1)  


        # output 
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out





