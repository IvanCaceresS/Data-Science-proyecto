import os
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# Definir los directorios base
base_dir = Path('./flower_photos')  # Directorio con las categorías originales
base_train_dir = Path('./flower_photos/train')
base_test_dir = Path('./flower_photos/test')
base_val_dir = Path('./flower_photos/val')

val_dir = Path(base_val_dir)
val_images = list(val_dir.glob('*/*.jpg')) #list of all images (full path)
print('\nCarpeta de validación: ', val_dir)
print('Número de imágenes conjunto validación: ', len(val_images))

data_dir = Path(base_train_dir)
folder = list(data_dir.glob('*'))
images = list(data_dir.glob('*/*.jpg')) #list of all images (full path)
print('\nCarpeta de entrenamiento: ', data_dir)
print('Número de imágenes conjunto entrenamiento: ', len(images))
print("-"*80,'\n')

image_size = 256
batch_size = 64

idg = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,       # Rotar las imágenes en un rango de hasta 40 grados
    width_shift_range=0.2,  # Desplazamientos horizontales
    height_shift_range=0.2, # Desplazamientos verticales
    shear_range=0.2,        # Corte de imágenes
    zoom_range=0.2,         # Zoom aleatorio
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',    # Rellenar píxeles que podrían quedar vacíos después de una transformación
)


train_gen = idg.flow_from_directory(base_train_dir,
                                    target_size=(image_size, image_size),
                                    class_mode='categorical',
                                    batch_size=batch_size,
                                    shuffle=True,
                                    seed=1
                                    )

val_gen = idg.flow_from_directory(base_val_dir,
                                  target_size=(image_size, image_size),
                                  class_mode='categorical',
                                  batch_size=batch_size,
                                  shuffle=True,
                                  seed=1
                                  )

#Classes train_gen
classes = train_gen.class_indices
print(classes)
class_names = []
for c in classes:
    class_names.append(c)
print('Los nombre de las clases de train son: ', class_names)

print('\n',"-"*80,'\n')

#Classes val_gen
classes = val_gen.class_indices
print(classes)
class_names = []
for c in classes:
    class_names.append(c)
print('Los nombre de las clases de val son: ', class_names)

print('\n',"-"*80,'\n')

#Numero de imagenes de entrenamiento
print("Número de batches de entrenamiento: ", len(train_gen))
#Numero de imagenes de validación
print("Número de batches de validación: ", len(val_gen))

print('\n',"-"*80,'\n')

model_path = './modelos/modelo_1000_epocas_64_batchsize_imagenes_iniciales_9969.keras'
if os.path.exists(model_path):
    print("Cargando modelo existente...")
    model = tf.keras.models.load_model(model_path)
    print("Modelo cargado.")
else:
    print("Entrenando nuevo modelo...")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(image_size,image_size,3))) # Input layer
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu')) # 2D Convolution layer
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) # Max Pool layer
    model.add(tf.keras.layers.BatchNormalization()) # Normalization layer
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu')) # 2D Convolution layer
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) # Max Pool layer
    model.add(tf.keras.layers.BatchNormalization()) # Normalization layer
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), strides=(1,1), activation='relu')) # 2D Convolution layer
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) # Max Pool layer
    model.add(tf.keras.layers.BatchNormalization()) # Normalization layer
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), strides=(1,1), activation='relu')) # 2D Convolution layer
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) # Max Pool layer
    model.add(tf.keras.layers.GlobalMaxPool2D()) # Global Max Pool layer
    model.add(tf.keras.layers.Flatten()) # Dense Layers after flattening the data
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2)) # Dropout
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization()) # Normalization layer
    model.add(tf.keras.layers.Dense(5, activation='softmax')) # Add Output Layer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.summary()

    #Imprime el número de imágenes de entrenamiento y validación
    print("Número de imágenes de entrenamiento: ", len(train_gen.filenames))
    print("Número de imágenes de validación: ", len(val_gen.filenames))

    #Imprime steps_per_epoch y validation_steps
    print("steps_per_epoch: ", len(train_gen.filenames) // batch_size)
    print("validation_steps: ", len(val_gen.filenames) // batch_size)

    print('\n',"-"*80,'\n')

    activity = model.fit(
        train_gen,
        epochs=1000,  # Número máximo de épocas
        steps_per_epoch = len(train_gen.filenames) // batch_size,
        validation_data=val_gen,
        validation_steps = len(val_gen.filenames) // batch_size,
        verbose=1
    )
    # Save the model
    model.save(model_path)

#PREDICCION DE 5 IMAGENES POR CADA CATEGORIA:

# Función para predecir una imagen
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (image_size, image_size))
    img = np.reshape(img, [1, image_size, image_size, 3])
    x = img / 255.0
    pred = model.predict(x)
    predicted_class = class_names[np.argmax(pred)]
    percentage = np.max(pred)
    return image_path, predicted_class, round(percentage * 100, 2)  # Devuelve también la ruta de la imagen

# Función para procesar imágenes por categoría y calcular estadísticas generales
def process_images_by_category(base_dir, num_images=5):
    categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    results = {}
    total_correct = 0
    total_images = 0
    for category in categories:
        correct_predictions = 0
        category_path = os.path.join(base_dir, category)
        images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith('.jpg')]
        selected_images = random.sample(images, min(num_images, len(images)))  # Selecciona 5 imágenes o menos si no hay suficientes
        predictions = [predict_image(img) for img in selected_images]
        for image_path, predicted_class, confidence in predictions:
            if predicted_class == category:
                correct_predictions += 1
        results[category] = {
            "predictions": predictions,
            "correct": correct_predictions,
            "total": len(selected_images)
        }
        total_correct += correct_predictions
        total_images += len(selected_images)
    overall_accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0
    return results, total_correct, total_images, overall_accuracy

# Ruta al directorio de test (para la predicción)
test_dir = './flower_photos/test'

# Ejecuta la predicción y obtiene estadísticas generales
predictions, total_correct, total_images, overall_accuracy = process_images_by_category(test_dir)
for category, data in predictions.items():
    print(f"Categoría: {category}")
    print(f"Predijo correctamente {data['correct']} de {data['total']} imágenes")
    for image_path, predicted_class, confidence in data['predictions']:
        print(f"Imagen: {image_path} - Predicha como {predicted_class} con una confianza del {confidence}%")

# Imprime el resumen final
print(f"\nTotal general: Predijo correctamente {total_correct} de {total_images} imágenes ({overall_accuracy:.2f}%)")