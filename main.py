import streamlit as st
import cv2
from transformers import ViTForImageClassification, ViTImageProcessor
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import timedelta
import yt_dlp


imagenet_classes = [
    "guillotine",
    "revolver",
    "rifle",
    "knife",
    "assault rifle",
    "chain saw",
    "torch",
    "sword",
    "missile",
    "bomb",
    "grenade",
    "tank",
    "crossbow",
    "bear trap",
    "cannon",
    "trident",
    "scissors",
    "machete",
    "dart",
    "ax"
]

# Paso 1: Función para descargar el video de YouTube
def download_video(youtube_url, output_path="video.mp4"):
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'best',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        get_info = ydl.extract_info(youtube_url, download=False) 
        st.write(f"Descargando el video: {get_info.get('title', None)} .....")
        ydl.download([youtube_url])
        st.write(f"Video descargado y guardado como: {output_path}")

# Paso 2: Cargar el modelo preentrenado de Hugging Face
def load_model():
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    return model, feature_extractor

# Paso 3: Procesar frames del video y detectar objetos
def detect_objects_in_video(video_path, model, feature_extractor, log_file="object_log.txt"):
    st.write(f"Iniciando el analisis...")
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    detected_objects = set()  # Para registrar objetos ya detectados

    with open(log_file, 'w') as log:
        while True:
            ret, frame = video.read()
            if not ret:
                break

            frame_count += 1
            # Extraer segundos exactos del frame actual y convertir a hh:mm:ss
            seconds = frame_count / fps
            time_format = str(timedelta(seconds=int(seconds)))

            # Convertir el frame a formato de entrada para el modelo
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = feature_extractor(images=img_rgb, return_tensors="pt")

            # Realizar la predicción
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = model.config.id2label[predicted_class_idx].lower()

            # Clasificar si es un objeto peligroso
            for label in imagenet_classes:
                if label in predicted_label and label not in detected_objects:
                    detected_objects.add(label)
                    log_entry = f"Objeto peligroso detectado: {model.config.id2label[predicted_class_idx]}, Tiempo: {time_format}"
                    log.write(log_entry + "\n")
                    log.flush()
                    

    video.release()
    st.write(f"El analisis a concluido. El Log puede encontrarse en {log_file}")
    return log_file

# Generar un mapa de palabras desde el archivo de log
def wordcloud_from_log(log_file):
    with open(log_file, 'r') as log:
        text = log.read()
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    return wordcloud

# Streamlit
st.title("Detección de Armas en Videos de YouTube")

# URL del video de YouTube
youtube_url = st.text_input("Ingresa la URL del video de YouTube para que sea analizado:")

if youtube_url:
    video_path = "video.mp4"

    if st.button("Descargar y Procesar el Video en fotogramas"):
        # Descargar el video
        download_video(youtube_url, video_path)
        # Cargar el modelo
        model, feature_extractor = load_model()
        # Detectar objetos en el video
        log_file = detect_objects_in_video(video_path, model, feature_extractor)
        with open(log_file, 'r') as log:
            log_content = log.readlines()

        if log_content:
            st.image("warning.jpg", caption="¡CUIDADO! Se detectaron armas en el video :( .", width=300)
            # Generar y mostrar el mapa de palabras
            st.write("Mapa de palabras: Armas detectadas en el video")
            wordcloud = wordcloud_from_log(log_file)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)
            st.write("******************** Contenido del Log ********************\n")
            st.code("".join(log_content), language="text")
        else:
            st.image("safe.jpg", caption="¡VIDEO APROPIADO! No se detectaron armas :) .", width=500)

    
        