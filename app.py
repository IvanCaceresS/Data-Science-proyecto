import os
import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk, Label, Canvas, Button, Toplevel, Frame, Scrollbar, VERTICAL, RIGHT, Y, BOTH
from tkinter import ttk
from tkinter import font as tkFont
from PIL import Image, ImageTk
import datetime

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
BRIGHTNESS_THRESHOLD_HIGH = 100
PREDICTION_DELAY = 50  # Delay de 50 ms entre predicciones

# Variables globales
continue_prediction = True
last_frame = None
last_results = None
prediction_saved = False
frame_status = ""

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
    #print(f'Brightness: {brightness}')
    return brightness < threshold

def is_frame_too_bright(frame, threshold=BRIGHTNESS_THRESHOLD_HIGH):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    #print(f'Brightness: {brightness}')
    return brightness > threshold

def show_prediction_results(results):
    global last_results, prediction_saved, frame_status
    last_results = results
    prediction_saved = False  # Reset the flag when new results are shown

    # Configurar el texto principal
    most_likely, confidence = results[0]
    result_text = f"Flor más probable: {most_likely} ({confidence * 100:.2f}%)"
    prediction_label.config(text=result_text, wraplength=600, anchor='w', justify='left')
    prediction_label.pack(pady=15)

    # Configurar el texto de las otras predicciones
    label_others.config(text="Pero también puede ser:", wraplength=600, anchor='w', justify='left')
    label_others.pack(pady=5)

    for idx in range(1, 3):
        predicted_class, confidence = results[idx]
        result_text = f"{predicted_class}: {confidence * 100:.2f}%"
        other_labels[idx - 1].config(text=result_text, wraplength=600, anchor='w', justify='left')
        other_labels[idx - 1].pack()

    frame_status = "normal"

def update_frame():
    global last_frame, frame_status
    if continue_prediction:
        ret, frame = cap.read()
        if ret:
            start_point = (FRAME_RECT[0], FRAME_RECT[1])
            end_point = (FRAME_RECT[2], FRAME_RECT[3])
            color = (255, 113, 82)
            thickness = 2
            cv2.rectangle(frame, start_point, end_point, color, thickness)
            frame = cv2.cvtColor(frame, BGR_TO_RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            canvas.imgtk = imgtk
            canvas.create_image(0, 0, anchor='nw', image=imgtk)
            
            if is_frame_dark(frame):
                prediction_label.config(text="Cámara tapada o muy poca luz. No se puede realizar la predicción.", wraplength=600, anchor='w', justify='left')
                prediction_label.pack(pady=10)
                label_others.pack_forget()
                for label in other_labels:
                    label.pack_forget()
                frame_status = "dark"
            elif is_frame_too_bright(frame):
                prediction_label.config(text="Demasiada luz. No se puede realizar la predicción.", wraplength=600, anchor='w', justify='left')
                prediction_label.pack(pady=10)
                label_others.pack_forget()
                for label in other_labels:
                    label.pack_forget()
                frame_status = "bright"
            else:
                crop_img = frame[FRAME_RECT[1]:FRAME_RECT[3], FRAME_RECT[0]:FRAME_RECT[2]]
                results = predict_image(crop_img, model, CLASS_NAMES)
                show_prediction_results(results)
                last_frame = crop_img

        root.after(PREDICTION_DELAY, update_frame)

def save_prediction():
    global prediction_saved
    if last_frame is not None and last_results is not None:
        if frame_status in ["dark", "bright"]:
            print("[INFO] No hay nada que guardar debido a condiciones de luz inadecuadas.")
            prediction_label.config(text="No hay nada que guardar debido a condiciones de luz inadecuadas.", wraplength=600, anchor='w', justify='left')
        elif not prediction_saved:
            most_likely, _ = last_results[0]
            now = datetime.datetime.now()
            timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
            dir_name = f"./predicciones/{most_likely}_{timestamp}"
            os.makedirs(dir_name, exist_ok=True)
            
            # Guardar la imagen
            img_path = os.path.join(dir_name, "imagen.jpg")
            cv2.imwrite(img_path, cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))
            
            # Guardar los resultados
            results_path = os.path.join(dir_name, "resultados.txt")
            with open(results_path, "w") as f:
                for result in last_results:
                    f.write(f"{result[0]}: {result[1] * 100:.2f}%\n")
            
            print(f"[INFO] Predicción guardada en {dir_name}")
            prediction_label.config(text="Predicción guardada.", wraplength=600, anchor='w', justify='left')
            prediction_saved = True
        else:
            print("[INFO] Predicción ya fue guardada.")
            prediction_label.config(text="Predicción ya fue guardada.", wraplength=600, anchor='w', justify='left')

def stop_prediction():
    global continue_prediction
    continue_prediction = False
    finalize_button.place(x=2800, y=2800)
    save_button.grid()
    restart_button.grid()
    view_predictions_button.grid()

def restart_prediction():
    global continue_prediction
    continue_prediction = True
    save_button.grid_remove()
    restart_button.grid_remove()
    view_predictions_button.grid_remove()
    finalize_button.place(x=280, y=544)
    finalize_button.lift()
    update_frame()

def view_predictions():
    pred_dir = './predicciones'
    if not os.path.exists(pred_dir):
        prediction_label.config(text="No hay predicciones guardadas.", wraplength=600, anchor='w', justify='left')
        return

    pred_window = Toplevel(root)
    pred_window.title("Predicciones Anteriores")
    pred_window.geometry("640x480")
    pred_window.resizable(False, False)
    pred_window.configure(bg='white')
    
    # Obtener las dimensiones de la pantalla
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Calcular la posición x e y para centrar la ventana
    position_right = int(screen_width/2 - 640/2)
    position_down = int(screen_height/2 - 480/2)
    
    # Establecer la posición de la ventana
    pred_window.geometry(f"640x480+{position_right}+{position_down}")
    
    # Cambiar el icono de la ventana
    icon_path = './icon.png'  # Asegúrate de que este archivo exista
    icon_img = ImageTk.PhotoImage(file=icon_path)
    pred_window.iconphoto(False, icon_img)
    
    frame = Frame(pred_window)
    frame.pack(fill=BOTH, expand=True)
    
    canvas = Canvas(frame)
    scrollbar = Scrollbar(frame, orient=VERTICAL, command=canvas.yview)
    scrollable_frame = Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    scrollbar.pack(side=RIGHT, fill=Y)
    canvas.pack(fill=BOTH, expand=True)
    
    def on_mouse_wheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    canvas.bind_all("<MouseWheel>", on_mouse_wheel)
    
    row = 0
    col = 0
    for folder in os.listdir(pred_dir):
        folder_path = os.path.join(pred_dir, folder)
        if os.path.isdir(folder_path):
            img_path = os.path.join(folder_path, "imagen.jpg")
            results_path = os.path.join(folder_path, "resultados.txt")
            
            if os.path.exists(img_path) and os.path.exists(results_path):
                img = Image.open(img_path)
                img.thumbnail((150, 150), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(img)
                
                img_label = Label(scrollable_frame, image=imgtk)
                img_label.image = imgtk
                img_label.grid(row=row, column=col, padx=30, pady=0)
                
                with open(results_path, "r") as f:
                    results_lines = f.readlines()
                    results_text = ''.join(results_lines[:3])  # Mostrar solo las tres primeras líneas
                
                text_label = Label(scrollable_frame, text=results_text, wraplength=200, anchor='center', justify='left')
                text_label.grid(row=row + 1, column=col, padx=0, pady=3)
                
                col += 1
                if col == 3:  # Si la columna actual es 3, reiniciar el contador de columnas y aumentar el contador de filas
                    col = 0
                    row += 2

model = load_model(MODEL_PATH)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("[ERROR] No se puede abrir la cámara")

root = Tk()
root.title("Clasificador de Flores")
root.geometry(f"+{int(root.winfo_screenwidth()/2 - 320)}+{int(root.winfo_screenheight()/2 - 340)}")
root.resizable(False, False)
root.configure(bg='white')

# Cambiar el icono de la ventana
icon_path = './icon.png'  # Asegúrate de que este archivo exista
icon_img = ImageTk.PhotoImage(file=icon_path)
root.iconphoto(False, icon_img)

canvas = Canvas(root, width=640, height=480, bg='white')
canvas.pack(pady=20)

prediction_label = Label(root, font=('Berlin Sans FB', 14), bg='white')
label_others = Label(root, font=('Berlin Sans FB', 12, 'bold'), bg='white')
other_labels = [Label(root, font=('Berlin Sans FB', 12), bg='white') for _ in range(2)]

finalize_button = Button(root, text="Finalizar", command=stop_prediction, font=('Berlin Sans FB', 14), bg='#041E42', fg='white')
#establecer la posicion x e y del boton
finalize_button.place(x=280, y=544)

# Crear un frame para contener los botones
button_frame = Frame(root, bg='white')
button_frame.pack(pady=20)

save_button = Button(button_frame, text="Guardar predicción", command=save_prediction, font=('Berlin Sans FB', 14), bg='#8CB3EA', fg='black')
save_button.grid(row=0, column=0, padx=5)

restart_button = Button(button_frame, text="Realizar otra predicción", command=restart_prediction, font=('Berlin Sans FB', 14), bg='#041E42', fg='white')
restart_button.grid(row=0, column=1, padx=5)

view_predictions_button = Button(button_frame, text="Ver Historial", command=view_predictions, font=('Berlin Sans FB', 14), bg='#8CB3EA', fg='black')
view_predictions_button.grid(row=0, column=2, padx=5)

save_button.grid_remove()
restart_button.grid_remove()
view_predictions_button.grid_remove()

print("[INFO] Iniciando actualización de frames...")
update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
print("[INFO] Recursos liberados, aplicación terminada.")
