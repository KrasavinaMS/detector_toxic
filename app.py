import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "MariaKras/toxic-model"  

st.set_page_config(page_title="Детектор токсичности", page_icon="🔍")
st.title("🔍 Детектор токсичных комментариев")

@st.cache_resource
def load_model():
    with st.spinner("Загрузка модели с Hugging Face..."):
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model.eval()
        return model, tokenizer

model, tokenizer = load_model()
st.success("✅ Модель загружена!")

text = st.text_area("✏️ Введите текст:", height=150)

if st.button("🔍 Проверить"):
    if text:
        encodings = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**encodings)
            prob = torch.nn.functional.softmax(outputs.logits, dim=-1)[0][1].item()
        
        if prob > 0.5:
            st.error(f"⚠️ ТОКСИЧНО (уверенность: {prob:.1%})")
        else:
            st.success(f"✅ НЕЙТРАЛЬНО (уверенность: {1-prob:.1%})")
