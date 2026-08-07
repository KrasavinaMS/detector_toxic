# Детектор токсичных комментариев

Сервис для определения токсичности текстов на русском языке на основе тонко настроенной модели **rubert-tiny2**.

## Модель

Обученная модель доступна для скачивания и использования на Hugging Face Hub:

👉 **[Ссылка на модель: `MariaKras/toxic-model`](https://huggingface.co/MariaKras/toxic-model)**

Модель автоматически загружается при запуске приложения.

## 🚀 Запуск приложения

### Локальный запуск

1.  **Клонируйте репозиторий:**
    '''bash
    git clone https://github.com/KrasavinaMS/detector_toxic.git
    cd detector_toxic
2. Установите зависимости:
bash
pip install -r requirements.txt

3. Запустите Streamlit-приложение:
bash
streamlit run app.py

4. Откройте приложение в браузере по адресу http://localhost:8501.
