import streamlit as st
from PIL import Image
import keras
import numpy as np
from keras.applications import VGG19
import requests
import os

def PreprocessAndPredict(image):
    # Определение списка названий классов
    class_names = ["Acne", "Eczema", "Atopic", "Psoriasis", "Tinea", "Vitiligo"]

    # Получение модели из репозитория GitHub
    model_url = 'https://github.com/Nemati213/Test/raw/main/6claass.h5'
    response = requests.get(model_url)

    # Проверка статуса запроса
    if response.status_code == 200:
        # Сохранение модели во временный файл
        with open("model.h5", "wb") as file:
            file.write(response.content)

        # Загрузка сохраненной модели
        model = keras.models.load_model("model.h5")
        vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(180, 180, 3))

        # Загрузка и предобработка изображения
        img = Image.open(image)
        img = img.resize((180, 180))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        img = vgg_model.predict(img)
        img = img.reshape(1, -1)

        # Предсказание на предобработанном изображении
        pred = model.predict(img)[0]
        predicted_class_index = np.argmax(pred)
        predicted_class_name = class_names[predicted_class_index]

        # Удаление временного файла модели
        os.remove("model.h5")

        return predicted_class_name
    else:
        st.write("Не удалось загрузить модель")
        return None

def load_image():
    """Создание формы для загрузки изображения"""
    # Форма для загрузки изображения
    uploaded_file = st.file_uploader(
        label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        # Получение загруженного изображения
        image = Image.open(uploaded_file)
        # Показ загруженного изображения на Web-странице
        st.image(image)
        image.save("img.jpg")

        # Возврат пути к изображению
        return "img.jpg"
    else:
        return None

# Вывод заголовка страницы
st.title('Классификатор кожных заболеваний')
# Вызов функции для загрузки изображения
img_path = load_image()

# Добавление кнопки-команды
if img_path:
    result = st.button("Распознать изображение")

    if result:
        st.write('**Результат распознования:**')
        st.write(PreprocessAndPredict(img_path))
