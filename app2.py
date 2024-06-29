import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_from_directory
from datetime import datetime
import base64
import hashlib

app = Flask(__name__)

CLASS_NAMES = ['Margarita', 'Diente de león', 'Rosas', 'Girasoles', 'Tulipanes']
CARE_INSTRUCTIONS = {
    'Margarita': '💧 Riego moderado, mantener el suelo húmedo pero no encharcado.\n☀️ Luz solar indirecta.\n🌱 Fertilizar mensualmente durante la temporada de crecimiento.',
    'Diente de león': '💧 Riego ligero, asegurándose de que el suelo se seque entre riegos.\n☀️ Pleno sol para un crecimiento óptimo.\n🌿 No requiere fertilización especial.',
    'Rosas': '💧 Riego frecuente, mantener el suelo uniformemente húmedo.\n☀️ Luz solar directa, al menos 6 horas diarias.\n✂️ Podar regularmente para promover un crecimiento saludable.\n🌱 Fertilizar cada 4-6 semanas durante la temporada de crecimiento.',
    'Girasoles': '💧 Riego abundante, especialmente durante el periodo de crecimiento.\n☀️ Pleno sol, requieren al menos 6-8 horas de luz solar al día.\n🌾 Suelo bien drenado y fértil.\n🌱 Fertilizar cada 2-3 semanas con un fertilizante balanceado.',
    'Tulipanes': '💧 Riego moderado, mantener el suelo húmedo pero bien drenado.\n☀️ Luz solar indirecta.\n🌷 Plantar los bulbos en otoño para una floración primaveral.\n🌱 Fertilizar con un fertilizante de bulbos al plantar y nuevamente cuando emerjan las hojas.'
}

MODEL_PATH = './modelos/modelo_resnet50_1000epocas_32batchsize_200patience.keras'
IMAGE_SIZE = (256, 256)

model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(image, model, class_names):
    img_array = np.array(image)
    img_array = cv2.resize(img_array, IMAGE_SIZE)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    predictions = model.predict(img_array)[0]
    top_indices = predictions.argsort()[-3:][::-1]
    results = [(class_names[i], predictions[i]) for i in top_indices]
    
    return results

def generate_dir_name(results):
    most_likely, _ = results[0]
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = f"{most_likely}_{timestamp}"
    return dir_name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = base64.b64decode(data['image'].split(',')[1])
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    
    results = predict_image(image, model, CLASS_NAMES)
    most_likely, _ = results[0]
    dir_name = generate_dir_name(results)
    
    os.makedirs(f"./predicciones/{dir_name}", exist_ok=True)
    
    img_path = os.path.join(f"./predicciones/{dir_name}", "imagen.jpg")
    cv2.imwrite(img_path, image)
    
    results_path = os.path.join(f"./predicciones/{dir_name}", "resultados.txt")
    with open(results_path, "w") as f:
        for result in results:
            f.write(f"{result[0]}: {result[1] * 100:.2f}%\n")
    
    others_list = [f"{r[0]}: {r[1] * 100:.2f}%" for r in results[1:]]
    
    response = {
        'prediction': f"Flor más probable: {most_likely} ({results[0][1] * 100:.2f}%)",
        'others': ["También puede ser:"] + others_list,
        'image': base64.b64encode(image_data).decode('utf-8')
    }
    
    return jsonify(response)


@app.route('/save_prediction', methods=['POST'])
def save_prediction():
    data = request.json
    image_data = base64.b64decode(data['image'])
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    
    results = predict_image(image, model, CLASS_NAMES)
    most_likely, _ = results[0]
    
    # Create a unique identifier for the prediction based on image data and results
    unique_id = hashlib.md5(image_data + str(results).encode()).hexdigest()
    dir_name = f"./predicciones/{most_likely}_{unique_id}"
    
    # Verify if a directory with the same unique_id exists
    if any(unique_id in folder for folder in os.listdir("./predicciones")):
        return jsonify({"message": "Predicción ya fue guardada."})
    
    os.makedirs(dir_name, exist_ok=True)
    
    img_path = os.path.join(dir_name, "imagen.jpg")
    cv2.imwrite(img_path, image)
    
    results_path = os.path.join(dir_name, "resultados.txt")
    with open(results_path, "w") as f:
        for result in results:
            f.write(f"{result[0]}: {result[1] * 100:.2f}%\n")
    
    return jsonify({"message": "Predicción guardada correctamente."})

@app.route('/view_predictions', methods=['GET'])
def view_predictions():
    predictions = []
    pred_dir = './predicciones'
    
    if os.path.exists(pred_dir):
        for folder in os.listdir(pred_dir):
            folder_path = os.path.join(pred_dir, folder)
            if os.path.isdir(folder_path):
                img_path = os.path.join(folder_path, "imagen.jpg")
                results_path = os.path.join(folder_path, "resultados.txt")
                
                if os.path.exists(img_path) and os.path.exists(results_path):
                    with open(results_path, "r") as f:
                        results_lines = f.readlines()
                    
                    predictions.append({
                        "image": f"/predicciones/{folder}/imagen.jpg",
                        "results": ''.join(results_lines[:3])
                    })
    
    return jsonify(predictions)

@app.route('/predicciones/<path:filename>')
def serve_image(filename):
    return send_from_directory('predicciones', filename)

@app.route('/view_cuidados', methods=['POST'])
def view_cuidados():
    data = request.json
    image_data = base64.b64decode(data['image'])
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    
    results = predict_image(image, model, CLASS_NAMES)
    most_likely, _ = results[0]
    
    cuidados = CARE_INSTRUCTIONS.get(most_likely, "No hay cuidados disponibles.")
    
    return jsonify({"cuidados": cuidados})

if __name__ == '__main__':
    app.run(debug=True)
