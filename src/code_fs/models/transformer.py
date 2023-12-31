import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads ==
                embed_size), "Embed size needs to be div by heads "

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.size(0)
        value_len, key_len, query_len = values.size(1), keys.size(1), queries.size(1)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        #Split into self.heads head
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        output = torch.einsum(
            "nhql, nlhd -> nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        output = self.fc_out(output)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()

        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        output = self.dropout(self.norm2(forward + x))
        return output


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, num_layers, embed_size, heads, forward_expansion, dropout, max_length, device):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_enbedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout,
                    forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.size()
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        output = self.dropout(
            self.word_enbedding(x) + self.position_embedding(positions)
        )
        for layer in self.layers:
            # values, key, query are all the same
            output = layer(output, output, output, mask)
        return output


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        mask_attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(mask_attention + x))
        output = self.transformer_block(value, key, query, src_mask)
        return output


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, num_layers, embed_size, heads, forward_expansion, dropout, max_length, device):
        super().__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size,
                    heads,
                    forward_expansion,
                    dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)

    def forward(self, x, encoder_out, src_mask, trg_mask):
        N, seq_len = x.size()
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.dropout(self.word_embedding(
            x) + self.position_embedding(positions))
        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, src_mask, trg_mask)
        output = self.fc_out(x)
        return output


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size=256,
                 num_layers=6,
                 forward_expansion=4,
                 heads=8,
                 dropout=0,
                 max_length=100,
                 device="cuda",
                 ):
        super().__init__()
        self.encoder = Encoder(src_vocab_size,
                               num_layers,
                               embed_size,
                               heads,
                               forward_expansion,
                               dropout,
                               max_length,
                               device)
        self.decoder = Decoder(trg_vocab_size,
                               num_layers,
                               embed_size,
                               heads,
                               forward_expansion,
                               dropout,
                               max_length,
                               device)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.size()
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        encoder_outout = self.encoder(src, src_mask)
        output = self.decoder(trg, encoder_outout, src_mask, trg_mask)
        return output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], 
                      [1, 8, 7, 3, 4, 5, 6, 7, 2]]
                    ).to(device)
    
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], 
                        [1, 5, 6, 2, 4, 7, 6, 2]]
                      ).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 100
    trg_vocab_size = 150
    model = Transformer(src_vocab_size, 
                        trg_vocab_size, 
                        src_pad_idx, 
                        trg_pad_idx, 
                        device=device)
    model.to(device)
    output = model(x, trg[:, :-1])
    print(output.size())