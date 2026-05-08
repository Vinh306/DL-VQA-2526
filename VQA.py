import torch.nn as nn
from CoAttention import CoAttention
from ImageEncoder import ImageEncoder
from QuestionEncoder import QuestionEncoder
from LSTMDecoder import LSTMDecoder
from TransformerDecoder import TransformerDecoder

class VQAModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim = 512,
        embed_dim = 256,
        decoder_type = "lstm",      # "lstm" | "transformer"
        phobert_name = "vinai/phobert-base",
        freeze_layers = 10,
        num_heads = 8,
        num_layers = 3
    ):
        super().__init__()
        IMG_DIM = 512    # ResNet18 output channels
        Q_DIM   = 768    # PhoBERT-base hidden size

        self.img_encoder  = ImageEncoder()
        self.q_encoder    = QuestionEncoder(phobert_name, freeze_layers) 
        self.co_attention = CoAttention(IMG_DIM, Q_DIM, hidden_dim)
        self.decoder_type = decoder_type

        if decoder_type == "lstm":
            self.decoder = LSTMDecoder(vocab_size, embed_dim, hidden_dim, enc_dim=hidden_dim)
        else:
            self.decoder = TransformerDecoder(vocab_size, hidden_dim, enc_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)

    def forward(self, image, q_input_ids, q_attn_mask, answer, teacher_forcing_ratio=0.5):
        # Encode
        img_feat = self.img_encoder(image)     # (B, P, 512)
        q_feat = self.q_encoder(q_input_ids, q_attn_mask)  # (B, L, 768)

        # Co-Attention Fusion
        img_ctx, q_out, q_vec = self.co_attention(img_feat, q_feat, q_attn_mask)

        # Decode
        if self.decoder_type == "lstm":
            return self.decoder(
                img_ctx, 
                q_out, 
                q_vec, 
                answer, 
                q_mask=q_attn_mask, 
                teacher_forcing_ratio=teacher_forcing_ratio
            )
        else:
            return self.decoder(img_ctx, q_out, q_vec, answer, q_mask=q_attn_mask)