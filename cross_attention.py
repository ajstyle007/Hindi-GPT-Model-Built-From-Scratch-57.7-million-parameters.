import torch
from torch import nn
import math

class Cross_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads  #head dim

        # Linear projections for Q, K, V
        # Linear Layers = Wq, Wk, Wv
        self.Q = nn.Linear(embed_dim, embed_dim)
        self.K = nn.Linear(embed_dim, embed_dim)
        self.V = nn.Linear(embed_dim, embed_dim)

        # Final linear layer (to combine heads)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q_decoder, K_encoder, V_encoder, mask=None):
        """
        Q_input = decoder hidden state  (B, S_dec, D)
        K_input = encoder output        (B, S_enc, D)
        V_input = encoder output        (B, S_enc, D)
        """
        B, S_dec, D = Q_decoder.size()
        # batch_size, seq_len, _ = x.size()
        S_enc = K_encoder.size(1)

        # If x = (batch, seq_len, embed_dim):
        # Linear projections
        Q = self.Q(Q_decoder)
        K = self.K(K_encoder)
        V = self.V(V_encoder)

        # Split into heads: [batch, seq_len, num_heads, head_dim] -> rearrange
        Q = Q.view(B, S_dec, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, S_enc, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, S_enc, self.num_heads, self.d_k).transpose(1, 2)

        # d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
                                        
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # Concatenate all heads and project back to embed_dim
        out = output.transpose(1,2).contiguous().view(B, S_dec, self.embed_dim)
        out = self.fc_out(out)

        return out