import torch 
from torch import nn
from positional_encoding import Positional_Encoding
from feed_forward_nn import feedforward
# from multihead_attention import MultiHeadAttention
from encoder_layer import Encoder_block
from masked_mha import Masked_MHA
from cross_attention import Cross_Attention

d_model = 512  # main model dimension
num_heads = 8  # number of heads
d_ff = 2048    # feedforward hidden dimension
seq_len = 128  # max input length
vocab_size = 30000


embedding_layer = nn.Embedding(vocab_size, d_model)
pos_encoding = Positional_Encoding(seq_len, d_model)

def prepare_encoder_input(token_ids):
    token_ids = torch.tensor(token_ids).unsqueeze(0)  # (1, seq_len)

    # 1. Convert token IDs → learned embeddings
    x = embedding_layer(token_ids)                      # (1, seq_len, d_model)

    # 2. Add sinusoidal positional encoding
    x = pos_encoding(x)                                 # (1, seq_len, d_model)

    return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, d_ff, num_heads):
        super().__init__()
        self.layers = nn.ModuleList(
            [Encoder_block(d_model, d_ff, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ✨ Why encoder-out is both K and V?
# Because:
# Keys represent what the decoder should “look at”
# Values represent what information the decoder can “take” from that position

# Encoder-out har token ka fully-processed semantic representation hota hai.
# Isi liye transformer decoder unhi representations ko attend karta hai translation/generation ke liye.


def generate_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    mask = (~mask).unsqueeze(0).unsqueeze(1)   # (1,1,L,L)
    return mask



class Decoder_Block(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()

        self.ffn = feedforward(d_model, d_ff)
        # self.multi_att = MultiHeadAttention(d_model, num_heads) #d_model >> embed_dim
        self.masked_mha = Masked_MHA(d_model, num_heads)
        self.cross_att = Cross_Attention(d_model, num_heads)
        self.norm_layer1 = nn.LayerNorm(d_model)
        self.norm_layer2 = nn.LayerNorm(d_model)
        self.norm_layer3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, mask=None, padding_mask=None):

        B, S, D = x.shape
        if mask is None:
            mask = generate_subsequent_mask(S)  # (1,1,S,S)
        # Masked Multi-Head Self Attention
        masked_mha_out = self.masked_mha(x, mask)

        # first Add & Norm (Residual connection)
        residual_1 = x + self.dropout(masked_mha_out)
        norm_layer1_out = self.norm_layer1(residual_1)

        # cross attention or encoder-decoder attention, this "out" is K, V coming from the encoder
        cross_att_out = self.cross_att(norm_layer1_out, enc_out, enc_out, mask=padding_mask)

        # second Add & Norm (Residual connection)
        residual_2 = norm_layer1_out + self.dropout(cross_att_out)
        norm_layer2_out = self.norm_layer2(residual_2)

        # Feed Forward Network
        ffn_out = self.ffn(norm_layer2_out)

        # third Add & Norm (Residual connection)
        residual_3 = norm_layer2_out + self.dropout(ffn_out)
        norm_layer3_out = self.norm_layer3(residual_3)

        return norm_layer3_out





def prepare_decoder_input(token_ids):
    token_ids = torch.tensor(token_ids).unsqueeze(0)  # (1, seq_len)

    # 1. Convert token IDs → learned embeddings
    x = embedding_layer(token_ids)                      # (1, seq_len, d_model)

    # 2. Add sinusoidal positional encoding
    x = pos_encoding(x)                                 # (1, seq_len, d_model)

    return x


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [Decoder_Block(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        # Original "Attention Is All You Need" paper did this
        # Har block ke baad tum already Add & Norm karte ho, lekin last block ke output me fir bhi thoda drift (distribution shift) aa jata hai.
        # Final LayerNorm output ko stabilize karta hai so that:
        # output distribution consistent ho
        # next layers (LM Head ya classifier) easily train ho
        # gradients stable rahe

    def forward(self, x, enc_out, mask=None):
        """
        x       : (B, S_dec, D)
        enc_out : (B, S_enc, D)
        tgt_mask: causal mask (1,1,S_dec,S_dec)
        """

        for layer in self.layers:
            x = layer(x, enc_out, mask)

        return self.norm(x)
    

# dec_in = prepare_decoder_input([7, 1542, 98])

enc_in = prepare_encoder_input([12,43,55,99])
encoder = Encoder(6, 512, 2048, 8)
enc_out = encoder(enc_in) # >> this out is both K, V will go to MHA attention(cross atten) of decoder

# decoder_block = Decoder_Block(512, 2048, 8)
# out = decoder_block(dec_in, enc_out)

# print("enc_out", enc_out.shape)
# print("decoder out", out.shape)

decoder = Decoder(num_layers=6, d_model=512, d_ff=2048, num_heads=8)

dec_inp = prepare_decoder_input([7, 1542, 98])    # (1, 3, 512)
enc_out = enc_out                                   # (1, 4, 512)

mask = generate_subsequent_mask(dec_inp.size(1))

out = decoder(dec_inp, enc_out, mask)
print(out.shape)  # torch.Size([1, 3, 512])