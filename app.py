import os
import time
import PIL
import cv2
import numpy as np
import tensorflow as tf

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

model_path = './modelos/modelo_resnet50_2000epocas_32_batchsize.keras'

if os.path.exists(model_path):
    print("Cargando modelo existente...: ", model_path)
    model = tf.keras.models.load_model(model_path)
    print("Modelo cargado.")
else:
    print("No existe el modelo en el directorio: ", model_path)
    # Fin del script
    exit()

def predict_image_from_cam(frame, model, class_names):
    # Redimensionar la imagen al tamaño de entrada esperado por el modelo
    img = cv2.resize(frame, (256, 256))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    
    # Realizar la predicción
    pred = model.predict(img)[0]
    
    # Obtener los tres principales índices de predicción y sus confianzas
    top_indices = pred.argsort()[-3:][::-1]  # Ordena y toma los 3 últimos índices
    results = [(class_names[i], pred[i]) for i in top_indices]
    
    return results

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("No se puede abrir la cámara")

text_height = 120
update_interval = 0.5  # Intervalo de actualización en segundos
last_update_time = time.time()
results_to_display = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        if current_time - last_update_time > update_interval:
            # Obtener las tres principales predicciones
            results_to_display = predict_image_from_cam(frame, model, class_names)
            print(results_to_display)
            last_update_time = current_time
        
        # Ajustar el frame para agregar espacio para el texto
        output_frame = np.zeros((frame.shape[0] + text_height, frame.shape[1], frame.shape[2]), dtype=np.uint8)
        output_frame[:frame.shape[0], :, :] = frame

        # Mostrar los resultados en el nuevo recuadro
        if results_to_display:
            for idx, (predicted_class, confidence) in enumerate(results_to_display):
                text = f"{predicted_class}: {confidence * 100:.2f}%"
                cv2.putText(output_frame, text, (10, frame.shape[0] + 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Mostrar el resultado
        cv2.imshow('Clasificador de flores', output_frame)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()