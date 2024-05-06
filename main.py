import numpy as np
import pandas as pd
import PIL
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import cv2
import shutil
from mpl_toolkits.axes_grid1 import ImageGrid

# Descargar el dataset y preparar el directorio
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True, cache_dir='.', cache_subdir='')
data_dir = pathlib.Path(data_dir)

# Crear los directorios para los conjuntos de datos
base_train_dir = data_dir / 'train'
base_test_dir = data_dir / 'test'
base_val_dir = data_dir / 'val'

# Crear directorios de base si no existen
for dir in (base_train_dir, base_test_dir, base_val_dir):
    if not dir.exists():
        os.makedirs(dir)

# Función para dividir los datos

def split_data(source, train_dir, test_dir, val_dir, train_size=0.8, test_size=0.10, val_size=0.10):
    # Validar que las proporciones sumen 1.0
    if train_size + test_size + val_size != 1.0:
        raise ValueError("La suma de train_size, test_size y val_size debe ser 1.0")
    
    # Recopilar archivos, omitiendo archivos ocultos y directorios
    files = [file for file in source.iterdir() if file.is_file() and not file.name.startswith('.')]
    np.random.shuffle(files)  # Mezclar los archivos para una división aleatoria
    files = np.array(files)
    
    # Calcular índices para dividir los archivos
    train, test, val = np.split(files, [int(train_size * len(files)), int((train_size + test_size) * len(files))])
    
    # Copiar archivos a los respectivos directorios
    for dir_path, file_set in zip((train_dir, test_dir, val_dir), (train, test, val)):
        os.makedirs(dir_path, exist_ok=True)
        for file in file_set:
            try:
                shutil.copy(file, dir_path)
            except IOError as e:
                print(f"No se pudo copiar {file} a {dir_path}. Error: {e}")

# Aplicación de la función a cada categoría
base_train_dir = data_dir / 'train'
base_test_dir = data_dir / 'test'
base_val_dir = data_dir / 'val'

for category in data_dir.iterdir():
    if category.is_dir() and category.name not in ['train', 'test', 'val']:
        category_name = category.name
        cat_train_dir = base_train_dir / category_name
        cat_test_dir = base_test_dir / category_name
        cat_val_dir = base_val_dir / category_name
        
        for dir in (cat_train_dir, cat_test_dir, cat_val_dir):
            os.makedirs(dir, exist_ok=True)
        
        split_data(category, cat_train_dir, cat_test_dir, cat_val_dir)

print("Datos divididos y almacenados correctamente.")
data_dir = pathlib.Path(base_train_dir)
folder = list(data_dir.glob('*'))
images = list(data_dir.glob('*/*.jpg')) #list of all images (full path)
print('Estructura de Carpetas:')
for f in folder:
    print(f)
print('\nNumber of images: ', len(images))

image_size = 256
batch_size = 32

idg = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

train_gen = idg.flow_from_directory(base_train_dir,
                                    target_size=(image_size, image_size),
                                    subset='training',
                                    class_mode='categorical',
                                    batch_size=batch_size,
                                    shuffle=True,
                                    seed=1
                                    )

val_gen = idg.flow_from_directory(base_val_dir,
                                  target_size=(image_size, image_size),                                                   
                                  subset='validation',
                                  class_mode='categorical',
                                  batch_size=batch_size,
                                  shuffle=True,
                                  seed=1
                                  )

classes = train_gen.class_indices
print(classes)
class_names = []
for c in classes:
    class_names.append(c)
print('Los nombre de las clases son: ', class_names)
 
model_path = './modelo_100_epocas.keras'
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
    model.summary()

    activity = model.fit(train_gen,
              epochs=100, # Increase number of epochs if you have sufficient hardware
              steps_per_epoch=1000//batch_size,  # Number of train images // batch_size
              validation_data=val_gen,
              validation_steps=10//batch_size, # Number of val images // batch_size
              verbose=1
    )
    # Save the model
    model.save(model_path)

# Predict a single image
img = cv2.imread('C:/Users/IVAN/Downloads/flor.jpeg')
img = cv2.resize(img, (256,256))
img = np.reshape(img, [1, 256, 256, 3])

model = tf.keras.models.load_model(model_path)
x = img/255.0
pred = model.predict(x)
print(pred)
# Considering the prediction is an array of probabilities, the class with the highest probability is selected
predicted_class = class_names[np.argmax(pred)]
percentage = np.max(pred)
print(predicted_class + ": " + str(round(percentage * 100,2)) + '%')
