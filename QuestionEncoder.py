import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class QuestionEncoder(nn.Module):
    def __init__(self, model_name="vinai/phobert-base", freeze_layer = 10):
        super().__init__()
        self.phobert  = AutoModel.from_pretrained(model_name)
        self.out_dim  = self.phobert.config.hidden_size # 768

        # Đóng băng N layer đầu để tiết kiệm bộ nhớ / tránh overfit
        if freeze_layer > 0:
            for i, layer in enumerate(self.phobert.encoder.layer):
                if i < freeze_layer:
                    for p in layer.parameters():
                        p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state: (B, L, 768)
        return out.last_hidden_state