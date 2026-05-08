import streamlit as st
import torch
import json
import os
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer, BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel
import text_utils
from VQA import VQAModel
from b1_zero_shot import ViEnTranslator
from deep_translator import GoogleTranslator

st.set_page_config(page_title="VQA Models Demo", page_icon="🖼️", layout="wide")

# CSS styling for beautiful UI
st.markdown("""
<style>
.metric-card {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    border-left: 5px solid #4CAF50;
    transition: transform 0.3s;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}
.model-title {
    color: #2c3e50;
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 10px;
}
.model-answer {
    color: #34495e;
    font-size: 1.1rem;
    font-style: italic;
}
[data-testid="stSidebar"] {
    background-color: #f1f3f6;
}
</style>
""", unsafe_allow_html=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT_TYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32

@st.cache_resource
def load_vocab(json_path="data/train.json"):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            train_samples = json.load(f)
        vocab = text_utils.Vocabulary.Vocabulary(freq_threshold=2)
        vocab.build_vocab([s["answer"] for s in train_samples])
        return vocab
    except Exception as e:
        st.error(f"Lỗi khi load từ điển: {e}")
        return None

@st.cache_resource
def load_custom_vqa(model_path, decoder_type, vocab_size):
    try:
        model = VQAModel(
            vocab_size=vocab_size,
            hidden_dim=512,
            embed_dim=256,
            decoder_type=decoder_type,
            phobert_name="vinai/phobert-base",
            freeze_layers=10,
            num_heads=8,
            num_layers=3
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Lỗi khi load mô hình {decoder_type}: {e}")
        return None

@st.cache_resource
def load_blip_zeroshot():
    from b1_zero_shot import BLIPB1ZeroShot
    try:
        model = BLIPB1ZeroShot(device=str(DEVICE))
        return model
    except Exception as e:
        st.error(f"Lỗi khi load BLIP ZeroShot: {e}")
        return None

@st.cache_resource
def load_blip_adapters():
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        base_model_1 = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base", torch_dtype=DT_TYPE)
        model_ft = PeftModel.from_pretrained(base_model_1, "lpv30/DL-CK").to(DEVICE).eval()
        
        base_model_2 = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base", torch_dtype=DT_TYPE)
        model_dpo = PeftModel.from_pretrained(base_model_2, "lpv30/DL-CK-DPO").to(DEVICE).eval()
        return processor, model_ft, model_dpo
    except Exception as e:
        st.error(f"Lỗi khi load BLIP LoRA adapters: {e}")
        return None, None, None

@st.cache_resource
def load_phobert_tokenizer():
    return AutoTokenizer.from_pretrained("vinai/phobert-base")

# Load global models
with st.spinner("Đang khởi tạo các module cơ sở..."):
    vocab = load_vocab()
    phobert_tokenizer = load_phobert_tokenizer()
    translator = ViEnTranslator()

with st.spinner("Đang tải toàn bộ 5 mô hình VQA lên bộ nhớ (Có thể tốn khoảng 1-2 phút tuỳ vào VRAM)..."):
    if vocab is not None:
        lstm_model = load_custom_vqa("checkpoints/best_lstm.pt", "lstm", len(vocab))
        transformer_model = load_custom_vqa("checkpoints/best_transformer.pt", "transformer", len(vocab))
    else:
        lstm_model = None
        transformer_model = None
        
    blip_zs = load_blip_zeroshot()
    blip_processor, blip_ft, blip_dpo = load_blip_adapters()

def transform_image_custom(image):
    transform = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def greedy_decode_lstm(model, imgs, q_ids, q_mask, sos_idx, eos_idx, max_len=20):
    B = imgs.size(0)
    img_feat = model.img_encoder(imgs)
    q_feat   = model.q_encoder(q_ids, q_mask)
    img_ctx, q_out, q_vec = model.co_attention(img_feat, q_feat, q_mask)

    decoder = model.decoder
    h = decoder.init_h(q_vec)
    c = decoder.init_c(q_vec)

    word = torch.full((B,), sos_idx, dtype=torch.long, device=DEVICE)
    done = [False] * B
    results = [[] for _ in range(B)]

    for _ in range(max_len):
        emb = decoder.embedding(word)
        img_context = decoder.attend_img(img_ctx, h)
        q_context   = decoder.attend_q(q_out, h, q_mask)

        lstm_input = torch.cat([emb, img_context, q_context], dim=-1)
        h, c = decoder.lstm_cell(lstm_input, (h, c))

        logits = decoder.fc_out(h)
        word   = logits.argmax(dim=-1)

        for i in range(B):
            if not done[i]:
                if word[i].item() == eos_idx:
                    done[i] = True
                else:
                    results[i].append(word[i].item())

        if all(done):
            break
    return results

@torch.no_grad()
def greedy_decode_transformer(model, imgs, q_ids, q_mask, sos_idx, eos_idx, max_len=20):
    B = imgs.size(0)
    img_feat = model.img_encoder(imgs)
    q_feat   = model.q_encoder(q_ids, q_mask)
    img_ctx, q_out, q_vec = model.co_attention(img_feat, q_feat, q_mask)

    dec_input = torch.full((B, 1), sos_idx, dtype=torch.long, device=DEVICE)
    done = [False] * B

    for _ in range(max_len):
        dummy = torch.zeros(B, 1, dtype=torch.long, device=DEVICE)
        inp   = torch.cat([dec_input, dummy], dim=1)

        logits    = model.decoder(img_ctx, q_out, q_vec, inp)
        next_tok  = logits[:, -1].argmax(dim=-1)

        dec_input = torch.cat([dec_input, next_tok.unsqueeze(1)], dim=1)

        for i in range(B):
            if next_tok[i].item() == eos_idx:
                done[i] = True
        if all(done):
            break

    results = []
    for i in range(B):
        toks = dec_input[i, 1:].tolist()
        if eos_idx in toks:
            toks = toks[:toks.index(eos_idx)]
        results.append(toks)
    return results

def ids_to_text(ids, vocab_obj):
    special = {vocab_obj.stoi[t] for t in ["<PAD>", "<SOS>", "<EOS>"]}
    tokens  = [vocab_obj.itos[i] for i in ids if i not in special]
    return " ".join(tokens)

# ---------- UI ----------
st.title("🤖 Ứng dụng Visual Question Answering (VQA)")
st.write("So sánh trực quan kết quả dự đoán từ 5 mô hình VQA khác nhau trên cùng một ảnh và câu hỏi.")
st.markdown("---")

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("1. Tải ảnh lên")
    uploaded_file = st.file_uploader("Chọn một bức ảnh (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        # Chuẩn hoá kích thước 384x384 để hiển thị
        display_image = image.resize((384, 384))
        st.image(display_image, caption="Ảnh 384x384 dùng cho dự đoán", use_container_width=True)
    else:
        image = None

with col2:
    st.subheader("2. Đặt câu hỏi")
    question_vi = st.text_input("Nhập câu hỏi tiếng Việt liên quan đến bức ảnh:", placeholder="Ví dụ: Bức ảnh này chụp những loại rau gì?")
    
    predict_clicked = st.button("🚀 Thực hiện Dự đoán", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.subheader("Kết quả dự đoán")
    
    # Placeholder for results
    result_container = st.container()
    
    if predict_clicked:
        if image is None:
            st.warning("Vui lòng tải ảnh lên trước khi dự đoán!")
        elif not question_vi.strip():
            st.warning("Vui lòng nhập câu hỏi!")
        else:
            with st.spinner("Đang tính toán các dự đoán..."):
                ans_lstm, ans_trans, ans_zs, ans_ft, ans_dpo = "Lỗi", "Lỗi", "Lỗi", "Lỗi", "Lỗi"
                
                # Custom LSTM & Transformer
                if vocab and lstm_model and transformer_model:
                    try:
                        q_enc = phobert_tokenizer(
                            question_vi, 
                            padding="max_length", 
                            max_length=128, 
                            truncation=True, 
                            return_tensors="pt"
                        ).to(DEVICE)
                        q_ids = q_enc["input_ids"]
                        q_mask = q_enc["attention_mask"]
                        
                        img_tensor = transform_image_custom(image)
                        
                        # Dự đoán LSTM
                        lstm_ids = greedy_decode_lstm(
                            lstm_model, img_tensor, q_ids, q_mask, 
                            sos_idx=vocab.stoi["<SOS>"], eos_idx=vocab.stoi["<EOS>"]
                        )
                        ans_lstm = ids_to_text(lstm_ids[0], vocab)
                        
                        # Dự đoán Transformer
                        trans_ids = greedy_decode_transformer(
                            transformer_model, img_tensor, q_ids, q_mask, 
                            sos_idx=vocab.stoi["<SOS>"], eos_idx=vocab.stoi["<EOS>"]
                        )
                        ans_trans = ids_to_text(trans_ids[0], vocab)
                    except Exception as e:
                        st.error(f"Lỗi khi chạy Custom Models: {e}")
                
                # BLIP Zero-shot
                if blip_zs:
                    try:
                        res_zs = blip_zs.predict(image, question_vi)
                        ans_zs = res_zs["answer_vi"]
                    except Exception as e:
                        st.error(f"Lỗi khi chạy BLIP ZeroShot: {e}")
                
                # BLIP Fine-tuned & DPO
                if blip_processor and blip_ft and blip_dpo:
                    try:
                        question_en = translator.to_english(question_vi)
                        blip_inputs = blip_processor(
                            images=image, text=question_en, return_tensors="pt"
                        ).to(DEVICE, DT_TYPE)
                        
                        # FT
                        out_ft = blip_ft.generate(**blip_inputs, max_new_tokens=50)
                        ans_en_ft = blip_processor.decode(out_ft[0], skip_special_tokens=True)
                        ans_ft = translator.to_vietnamese(ans_en_ft)
                        
                        # DPO
                        out_dpo = blip_dpo.generate(**blip_inputs, max_new_tokens=50)
                        ans_en_dpo = blip_processor.decode(out_dpo[0], skip_special_tokens=True)
                        ans_dpo = translator.to_vietnamese(ans_en_dpo)
                    except Exception as e:
                        st.error(f"Lỗi khi chạy BLIP LoRA Models: {e}")

            # Hiển thị
            def display_metric_card(title, value, color_border):
                return f"""
                <div class="metric-card" style="border-left-color: {color_border};">
                    <div class="model-title">{title}</div>
                    <div class="model-answer">{value}</div>
                </div>
                """
            
            with result_container:
                st.markdown(display_metric_card("LSTM Decoder", ans_lstm, "#FF5722"), unsafe_allow_html=True)
                st.markdown(display_metric_card("Transformer Decoder", ans_trans, "#2196F3"), unsafe_allow_html=True)
                st.markdown(display_metric_card("BLIP Zero-Shot", ans_zs, "#9C27B0"), unsafe_allow_html=True)
                st.markdown(display_metric_card("BLIP Fine-Tuned", ans_ft, "#FFEB3B"), unsafe_allow_html=True)
                st.markdown(display_metric_card("BLIP DPO", ans_dpo, "#4CAF50"), unsafe_allow_html=True)
