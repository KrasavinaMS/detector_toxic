import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Модерация токсичных комментариев", page_icon="🔍")
st.title("🔍 Детектор токсичных комментариев")
st.write("Введите текст, и модель определит, является ли он токсичным.")

@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained('toxic_model')
    tokenizer = AutoTokenizer.from_pretrained('toxic_model')
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

user_input = st.text_area("✏️ Ваш текст:", height=150)

if st.button("🔍 Проверить", type="primary"):
    if user_input:
        with st.spinner("Анализируем текст..."):
            encodings = tokenizer(user_input, truncation=True, padding=True, max_length=128, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**encodings)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            toxic_prob = probs[0][1].item()
            
            # Красивое отображение
            col1, col2 = st.columns(2)
            with col1:
                if toxic_prob > 0.5:
                    st.error("⚠️ ТОКСИЧНО")
                else:
                    st.success("✅ НЕЙТРАЛЬНО")
            with col2:
                st.metric("Уверенность", f"{toxic_prob:.1%}")
            
            # Прогресс-бар для наглядности
            st.progress(toxic_prob)
            st.caption(f"Вероятность токсичности: {toxic_prob:.2%}")
    else:
        st.warning("Пожалуйста, введите текст для проверки.")

# Добавляем примеры
with st.expander("📝 Примеры для тестирования"):
    st.write("Нажмите на пример, чтобы вставить его в поле ввода:")
    examples = [
        "Это отличный фильм, я в восторге!",
        "Ты идиот, что ты несешь?",
        "Сегодня хорошая погода",
        "Урод, закрой свой рот",
        "Спасибо за помощь!"
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.input_text = ex
            st.rerun()

# Если есть сохраненный текст в session_state, вставляем его
if "input_text" in st.session_state and st.session_state.input_text:
    st.text_area("✏️ Ваш текст:", value=st.session_state.input_text, height=150, key="text_input")
