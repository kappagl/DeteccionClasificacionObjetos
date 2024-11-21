from pytube import YouTube
import cv2
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from moviepy.editor import VideoFileClip
import os
import yt_dlp
from datetime import timedelta

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
        ydl.download([youtube_url])
        print(f"Video descargado: {output_path}")

# Paso 2: Cargar el modelo preentrenado de Hugging Face
def load_model():
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    return model, feature_extractor

# Paso 3: Procesar frames del video y detectar objetos
def detect_objects_in_video(video_path, model, feature_extractor, log_file="object_log.txt"):
    print(f"Análisis iniciado...")
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    detected_objects = set()  # Para mantener un registro de objetos ya detectados

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

            # Clasificar si es un objeto peligroso (ejemplo de objetos a verificar)
            for label in imagenet_classes:
                if label in predicted_label and label not in detected_objects:
                    detected_objects.add(label)
                    log.write(f"Objeto peligroso detectado: {model.config.id2label[predicted_class_idx]}, Tiempo: {time_format}\n")
                    print(f"Objeto peligroso detectado: {model.config.id2label[predicted_class_idx]}, Tiempo: {time_format}")

    video.release()
    print(f"Análisis completado. Log guardado en {log_file}")

# Ejecución del proceso completo
if __name__ == "__main__":
    # URL del video de YouTube
    youtube_url = "https://www.youtube.com/watch?v=e5KMHHXLzIw"
    video_path = "video.mp4"
    
    # Descargar el video
    download_video(youtube_url, video_path)

    # Cargar el modelo
    model, feature_extractor = load_model()

    # Detectar objetos en el video
    detect_objects_in_video(video_path, model, feature_extractor)