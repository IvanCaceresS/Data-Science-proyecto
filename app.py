import os
import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk, Button, Label, Canvas, Toplevel
from PIL import Image, ImageTk

# Configuración de la ruta del modelo y nombres de las clases en español
class_names = ['Margarita', 'Diente de león', 'Rosas', 'Girasoles', 'Tulipanes']
care_instructions = {
    'Margarita': 'Riego moderado, mantener el suelo húmedo pero no encharcado. Luz solar indirecta. Fertilizar mensualmente durante la temporada de crecimiento.',
    'Diente de león': 'Riego ligero, asegurándose de que el suelo se seque entre riegos. Pleno sol para un crecimiento óptimo. No requiere fertilización especial.',
    'Rosas': 'Riego frecuente, mantener el suelo uniformemente húmedo. Luz solar directa, al menos 6 horas diarias. Podar regularmente para promover un crecimiento saludable. Fertilizar cada 4-6 semanas durante la temporada de crecimiento.',
    'Girasoles': 'Riego abundante, especialmente durante el periodo de crecimiento. Pleno sol, requieren al menos 6-8 horas de luz solar al día. Suelo bien drenado y fértil. Fertilizar cada 2-3 semanas con un fertilizante balanceado.',
    'Tulipanes': 'Riego moderado, mantener el suelo húmedo pero bien drenado. Luz solar indirecta. Plantar los bulbos en otoño para una floración primaveral. Fertilizar con un fertilizante de bulbos al plantar y nuevamente cuando emerjan las hojas.'
}

model_path = './modelos/modelo_resnet50_2000epocas_32_batchsize.keras'

# Cargar el modelo si existe
if os.path.exists(model_path):
    print("Cargando modelo existente...: ", model_path)
    model = tf.keras.models.load_model(model_path)
    print("Modelo cargado.")
else:
    print("No existe el modelo en el directorio: ", model_path)
    exit()

def predict_image(frame, model, class_names):
    # Redimensionar la imagen al tamaño de entrada esperado por el modelo
    img = cv2.resize(frame, (256, 256))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    
    # Realizar la predicción
    pred = model.predict(img)[0]
    
    # Obtener los tres principales índices de predicción y sus confianzas
    top_indices = pred.argsort()[-3:][::-1]
    results = [(class_names[i], pred[i]) for i in top_indices]
    
    return results

def show_prediction_results(results, cropped_img):
    # Convertir la imagen recortada de BGR a RGB
    cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    
    # Crear una nueva ventana para mostrar los resultados
    result_window = Toplevel(root)
    result_window.title("Resultados de la Predicción")
    
    # Centrar la ventana en la pantalla y hacer que no se pueda redimensionar
    result_window.geometry(f"+{int(root.winfo_screenwidth()/2 - 200)}+{int(root.winfo_screenheight()/2 - 200)}")
    result_window.resizable(False, False)
    
    # Estilo de la ventana de resultados
    result_window.configure(bg='#F5F5F5')
    
    # Mostrar la imagen capturada
    img = Image.fromarray(cropped_img_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    img_label = Label(result_window, image=imgtk, bg='#F5F5F5')
    img_label.image = imgtk
    img_label.pack(pady=10)
    
    # Mostrar el resultado más probable y sus cuidados
    most_likely, confidence = results[0]
    result_text = f"Más probable: {most_likely} ({confidence * 100:.2f}%)\nCuidados: {care_instructions[most_likely]}"
    label = Label(result_window, text=result_text, font=("Arial", 14), bg='#F5F5F5', wraplength=500, justify='left')
    label.pack(pady=10)
    
    # Mostrar las dos categorías más probables que le siguen
    label_others = Label(result_window, text="También puede ser:", font=("Arial", 12, 'bold'), bg='#F5F5F5')
    label_others.pack(pady=5)
    
    for idx in range(1, 3):
        predicted_class, confidence = results[idx]
        result_text = f"{predicted_class}: {confidence * 100:.2f}%"
        label = Label(result_window, text=result_text, font=("Arial", 12), bg='#F5F5F5')
        label.pack()
    
    # Ejecutar la ventana de resultados
    result_window.mainloop()

def update_frame():
    ret, frame = cap.read()
    if ret:
        # Dibujar el cuadro en el frame
        start_point = (200, 100)
        end_point = (450, 350)
        color = (255, 0, 0)
        thickness = 2
        cv2.rectangle(frame, start_point, end_point, color, thickness)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.imgtk = imgtk
        canvas.create_image(0, 0, anchor='nw', image=imgtk)
    root.after(10, update_frame)

def capture_image():
    ret, frame = cap.read()
    if ret:
        # Recortar la imagen dentro del cuadro
        crop_img = frame[100:350, 200:450]
        
        # Realizar la predicción
        results = predict_image(crop_img, model, class_names)
        
        # Mostrar los resultados de la predicción junto con la imagen capturada
        show_prediction_results(results, crop_img)

# Configuración de la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("No se puede abrir la cámara")

# Crear la ventana principal
root = Tk()
root.title("Clasificador de Flores")

# Centrar la ventana en la pantalla y hacer que no se pueda redimensionar
root.geometry(f"+{int(root.winfo_screenwidth()/2 - 320)}+{int(root.winfo_screenheight()/2 - 240)}")
root.resizable(False, False)

# Estilo de la ventana principal
root.configure(bg='#E0E0E0')

# Canvas para mostrar la vista de la cámara
canvas = Canvas(root, width=640, height=480, bg='#E0E0E0')
canvas.pack(pady=20)

# Botón para capturar imagen
capture_button = Button(root, text="¡Quiero saber que flor es!", command=capture_image, font=("Arial", 14), bg='#B0C4DE', fg='black')
capture_button.pack(pady=20)

# Actualizar el frame de la cámara en la ventana
update_frame()

# Mostrar la ventana principal
root.mainloop()

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
