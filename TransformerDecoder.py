import torch
import torch.nn as nn
import torch.nn.functional as F
import math



# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 512, dropout = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()

        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, enc_dim, num_heads=8, num_layers=4, ff_dim=1024, dropout=0.1):
        super().__init__()

        assert hidden_dim % num_heads == 0

        self.vocab_size = vocab_size
        self.d_model    = hidden_dim

        # Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_enc   = PositionalEncoding(hidden_dim, dropout=dropout)

        # Memory projection
        self.mem_proj = nn.Linear(enc_dim, hidden_dim)
        self.q_proj = nn.Linear(enc_dim, hidden_dim)

        # Inject question
        self.q_inject = nn.Linear(hidden_dim, hidden_dim)

        self.q_gate = nn.Linear(hidden_dim, hidden_dim) # Thêm gate cho question

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.tf_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.fc_out  = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    # Forward 
    def forward(self, img_ctx, q_out, q_vec, answer, img_mask=None, q_mask=None):
        tgt_in = answer[:, :-1]  # (B, T-1)
        B, T = tgt_in.size()
        device = answer.device

        # Memory
        memory = torch.cat([self.mem_proj(img_ctx), self.q_proj(q_out)], dim=1)  # (B, P+L, D)  # (B, P, D)

        # Inject question
        q_bias = self.q_inject(q_vec).unsqueeze(1)  # (B, 1, D)
        memory = self.dropout(memory + q_bias)

        # Target embedding
        scale = math.sqrt(self.d_model)
        tgt = self.embedding(tgt_in) * scale
        tgt = self.pos_enc(tgt)

        q_gate = torch.sigmoid(self.q_gate(q_vec)).unsqueeze(1)  # (B, 1, D)
        tgt = tgt * q_gate

        # Mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        # Padding mask cho target
        tgt_pad_mask = (tgt_in == 0) 

        # Padding mask cho memory
        memory_pad_mask = None
        if img_mask is not None:
            memory_pad_mask = (img_mask == 0)

        # Decode
        decoded = self.tf_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask
        )

        logits = self.fc_out(self.dropout(decoded))
        return logits  # (B, T-1, vocab_size)
