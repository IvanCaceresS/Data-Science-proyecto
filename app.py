import os
import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk, Label, Canvas, Button
from PIL import Image, ImageTk

# Configuración de la ruta del modelo y nombres de las clases en español
CLASS_NAMES = ['Margarita', 'Diente de león', 'Rosas', 'Girasoles', 'Tulipanes']
CARE_INSTRUCTIONS = {
    'Margarita': 'Riego moderado, mantener el suelo húmedo pero no encharcado. Luz solar indirecta. Fertilizar mensualmente durante la temporada de crecimiento.',
    'Diente de león': 'Riego ligero, asegurándose de que el suelo se seque entre riegos. Pleno sol para un crecimiento óptimo. No requiere fertilización especial.',
    'Rosas': 'Riego frecuente, mantener el suelo uniformemente húmedo. Luz solar directa, al menos 6 horas diarias. Podar regularmente para promover un crecimiento saludable. Fertilizar cada 4-6 semanas durante la temporada de crecimiento.',
    'Girasoles': 'Riego abundante, especialmente durante el periodo de crecimiento. Pleno sol, requieren al menos 6-8 horas de luz solar al día. Suelo bien drenado y fértil. Fertilizar cada 2-3 semanas con un fertilizante balanceado.',
    'Tulipanes': 'Riego moderado, mantener el suelo húmedo pero bien drenado. Luz solar indirecta. Plantar los bulbos en otoño para una floración primaveral. Fertilizar con un fertilizante de bulbos al plantar y nuevamente cuando emerjan las hojas.'
}

MODEL_PATH = './modelos/modelo_resnet50_1000epocas_32batchsize_200patience.keras'
IMAGE_SIZE = (256, 256)
FRAME_RECT = (200, 100, 450, 350)
BGR_TO_RGB = cv2.COLOR_BGR2RGB
BRIGHTNESS_THRESHOLD_LOW = 30
BRIGHTNESS_THRESHOLD_HIGH = 180

# Variable global para controlar la actualización de frames
continue_prediction = True

def load_model(path):
    if os.path.exists(path):
        print(f"[INFO] Cargando modelo existente: {path}")
        return tf.keras.models.load_model(path)
    else:
        raise FileNotFoundError(f"[ERROR] No existe el modelo en el directorio: {path}")

def predict_image(frame, model, class_names):
    img = cv2.resize(frame, IMAGE_SIZE)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    
    pred = model.predict(img)[0]
    
    top_indices = pred.argsort()[-3:][::-1]
    results = [(class_names[i], pred[i]) for i in top_indices]
    
    return results

def is_frame_dark(frame, threshold=BRIGHTNESS_THRESHOLD_LOW):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness < threshold

def is_frame_too_bright(frame, threshold=BRIGHTNESS_THRESHOLD_HIGH):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness > threshold

def show_prediction_results(results):
    # Configurar el texto principal
    most_likely, confidence = results[0]
    result_text = f"Más probable: {most_likely} ({confidence * 100:.2f}%)\nCuidados: {CARE_INSTRUCTIONS[most_likely]}"
    prediction_label.config(text=result_text, wraplength=600, anchor='w', justify='left')
    prediction_label.pack(pady=10)

    # Configurar el texto de las otras predicciones
    label_others.config(text="También puede ser:", wraplength=600, anchor='w', justify='left')
    label_others.pack(pady=5)

    for idx in range(1, 3):
        predicted_class, confidence = results[idx]
        result_text = f"{predicted_class}: {confidence * 100:.2f}%"
        other_labels[idx - 1].config(text=result_text, wraplength=600, anchor='w', justify='left')
        other_labels[idx - 1].pack()

def update_frame():
    if continue_prediction:
        ret, frame = cap.read()
        if ret:
            start_point = (FRAME_RECT[0], FRAME_RECT[1])
            end_point = (FRAME_RECT[2], FRAME_RECT[3])
            color = (255, 0, 0)
            thickness = 2
            cv2.rectangle(frame, start_point, end_point, color, thickness)
            frame = cv2.cvtColor(frame, BGR_TO_RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            canvas.imgtk = imgtk
            canvas.create_image(0, 0, anchor='nw', image=imgtk)
            
            if is_frame_dark(frame):
                prediction_label.config(text="Cámara tapada. No se puede realizar la predicción.", wraplength=600, anchor='w', justify='left')
                prediction_label.pack(pady=10)
                label_others.pack_forget()
                for label in other_labels:
                    label.pack_forget()
            elif is_frame_too_bright(frame):
                prediction_label.config(text="Demasiada luz. No se puede realizar la predicción.", wraplength=600, anchor='w', justify='left')
                prediction_label.pack(pady=10)
                label_others.pack_forget()
                for label in other_labels:
                    label.pack_forget()
            else:
                crop_img = frame[FRAME_RECT[1]:FRAME_RECT[3], FRAME_RECT[0]:FRAME_RECT[2]]
                results = predict_image(crop_img, model, CLASS_NAMES)
                show_prediction_results(results)
                
        root.after(100, update_frame)

def stop_prediction():
    global continue_prediction
    continue_prediction = False
    finalize_button.pack_forget()
    restart_button.pack(pady=20)

def restart_prediction():
    global continue_prediction
    continue_prediction = True
    restart_button.pack_forget()
    finalize_button.pack(pady=20)
    update_frame()

model = load_model(MODEL_PATH)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("[ERROR] No se puede abrir la cámara")

root = Tk()
root.title("Clasificador de Flores")
root.geometry(f"+{int(root.winfo_screenwidth()/2 - 320)}+{int(root.winfo_screenheight()/2 - 240)}")
root.resizable(False, False)
root.configure(bg='#E0E0E0')

canvas = Canvas(root, width=640, height=480, bg='#E0E0E0')
canvas.pack(pady=20)

prediction_label = Label(root, font=("Arial", 14), bg='#E0E0E0')
label_others = Label(root, font=("Arial", 12, 'bold'), bg='#E0E0E0')
other_labels = [Label(root, font=("Arial", 12), bg='#E0E0E0') for _ in range(2)]

finalize_button = Button(root, text="Finalizar", command=stop_prediction, font=("Arial", 14), bg='#B0C4DE', fg='black')
finalize_button.pack(pady=20)

restart_button = Button(root, text="Realizar otra predicción", command=restart_prediction, font=("Arial", 14), bg='#B0C4DE', fg='black')
restart_button.pack_forget()

print("[INFO] Iniciando actualización de frames...")
update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
print("[INFO] Recursos liberados, aplicación terminada.")