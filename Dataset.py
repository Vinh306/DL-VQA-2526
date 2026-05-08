import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from text_utils import TextProcessor, Vocabulary
import os
from PIL import Image

class ViVQADataset(Dataset):
    def __init__(self, samples, image_root, vocabulary, transform=None):
        self.samples = samples
        self.image_root = image_root
        self.vocab = vocabulary
        self.transform  = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Ảnh
        rel  = item["image_path"].replace("\\", os.sep)
        img  = Image.open(os.path.join(self.image_root, rel)).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Câu hỏi: trả về string thô, PhoBERT tokenize trong collate
        question_text = item["question"]

        # Câu trả lời cho decoder
        ans_ids = (
            [self.vocab.stoi["<SOS>"]]
            + self.vocab.numericalize(item["answer"])
            + [self.vocab.stoi["<EOS>"]]
        )

        return img, question_text, torch.tensor(ans_ids)
    

class VQACollate:
    def __init__(self, tokenizer, pad_idx, max_q_len = 128):
        self.tokenizer = tokenizer
        self.pad_idx = pad_idx
        self.max_q_len = max_q_len

    def __call__(self, batch):
        imgs, q_texts, answers = zip(*batch)
        imgs = torch.stack(imgs)

        q_segmented = [" ".join(TextProcessor.tokenize(q)) for q in q_texts]

        # PhoBERT tokenize 
        encoded = self.tokenizer(
            q_segmented,
            padding=True,
            truncation=True,
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        q_input_ids = encoded["input_ids"]       # (B, L)
        q_attn_mask = encoded["attention_mask"]  # (B, L)

        answers = pad_sequence(answers, batch_first=True, padding_value=self.pad_idx)

        return imgs, q_input_ids, q_attn_mask, answers