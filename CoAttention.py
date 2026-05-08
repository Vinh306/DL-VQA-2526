import torch
import torch.nn as nn
import torch.nn.functional as F

class CoAttention(nn.Module):
    def __init__(self, img_dim, q_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim 
        self.scale = hidden_dim ** -0.5 # sqrt(d_k)
        # Projection cho image và question về cùng dimension để tính affinity
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        # LayerNorm để ổn định training sau khi fuse context và original feature
        self.norm_img = nn.LayerNorm(hidden_dim)
        self.norm_q = nn.LayerNorm(hidden_dim)
        # Gating để chọn lọc thông tin từ context
        self.gate_img = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.gate_q = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, img_feat, q_feat, q_mask=None):
        # Projection
        img_h = self.img_proj(img_feat)   # (B, P, D)
        q_h   = self.q_proj(q_feat)       # (B, L, D)

        # Affinity
        aff = torch.bmm(img_h, q_h.transpose(1, 2)) * self.scale  # (B, P, L)

        # Mask
        if q_mask is not None:
            pad_mask = (q_mask == 0).unsqueeze(1)  # (B, 1, L)
            aff = aff.masked_fill(pad_mask, -1e9)

        # Image attends Question 
        img_attn = self.dropout(F.softmax(aff, dim=-1))
        img_ctx  = torch.bmm(img_attn, q_h)

        # Question attends Image 
        q_attn = self.dropout(F.softmax(aff.transpose(1, 2), dim=-1))
        q_ctx  = torch.bmm(q_attn, img_h)

        # Gating + Residual
        g_img = self.gate_img(torch.cat([img_h, img_ctx], dim=-1))
        img_out = self.norm_img(img_h + g_img * img_ctx)

        g_q = self.gate_q(torch.cat([q_h, q_ctx], dim=-1))
        q_out = self.norm_q(q_h + g_q * q_ctx)

        # Pool question (cho LSTM)
        if q_mask is not None:
            mask = q_mask.unsqueeze(-1)
            q_vec = (q_out * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            q_vec = q_out.mean(dim=1)

        return img_out, q_out, q_vec