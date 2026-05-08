import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMDecoder(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, enc_dim, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Attention cho IMAGE
        self.W_img = nn.Linear(enc_dim, hidden_dim)
        self.W_h_img = nn.Linear(hidden_dim, hidden_dim) # <-- Đổi tên
        self.v_img = nn.Linear(hidden_dim, 1)

        # Attention cho QUESTION
        self.W_q = nn.Linear(enc_dim, hidden_dim)
        self.W_h_q = nn.Linear(hidden_dim, hidden_dim)   # <-- Thêm mới
        self.v_q = nn.Linear(hidden_dim, 1)

        # LSTM
        self.lstm_cell = nn.LSTMCell(embed_dim + enc_dim * 2, hidden_dim)

        # tạo hidden state từ q_vec
        self.init_h = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.init_c = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Output
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    # Attention image
    def attend_img(self, img_ctx, h, img_mask=None):
        # (B, P, D) → (B, P, H)
        e = self.v_img(torch.tanh(
            self.W_img(img_ctx) + self.W_h_img(h).unsqueeze(1)
        )).squeeze(-1)  # (B, P)

        if img_mask is not None:
            e = e.masked_fill(img_mask == 0, -1e9)

        alpha = F.softmax(e, dim=-1)
        ctx = torch.bmm(alpha.unsqueeze(1), img_ctx).squeeze(1)

        return ctx

    # Attention QUESTION
    def attend_q(self, q_out, h, q_mask=None):
        e = self.v_q(torch.tanh(
            self.W_q(q_out) + self.W_h_q(h).unsqueeze(1)
        )).squeeze(-1)  # (B, L)

        if q_mask is not None:
            e = e.masked_fill(q_mask == 0, -1e9)

        alpha = F.softmax(e, dim=-1)
        ctx = torch.bmm(alpha.unsqueeze(1), q_out).squeeze(1)

        return ctx

    # Forward
    def forward(self, img_ctx, q_out, q_vec, answer, img_mask=None, q_mask=None, teacher_forcing_ratio=0.5):
        B, T = answer.size()
        device = img_ctx.device

        output = torch.zeros(B, T, self.fc_out.out_features, device=device)

        # Khởi tạo hidden
        h = self.init_h(q_vec)
        c = self.init_c(q_vec)

        # <SOS>
        word = answer[:, 0]

        for t in range(1, T):

            emb = self.dropout(self.embedding(word))

            # Attention 
            img_context = self.attend_img(img_ctx, h, img_mask)
            q_context = self.attend_q(q_out, h, q_mask)

            # LSTM input
            lstm_input = torch.cat([emb, img_context, q_context], dim=-1)

            h, c = self.lstm_cell(lstm_input, (h, c))

            logits = self.fc_out(self.dropout(h))
            output[:, t] = logits

            # Teacher forcing 
            use_tf = torch.rand(1, device=device).item() < teacher_forcing_ratio
            if use_tf:
                word = answer[:, t]
            else:
                word = logits.argmax(dim=-1)

        return output