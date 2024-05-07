import os
import random
import pathlib
import shutil
import PIL

import cv2
import numpy as np
import tensorflow as tf

def setup_dataset(url, path='./flower_photos'):
    if os.path.exists(path):
        print("Eliminando la carpeta flower_photos")
        shutil.rmtree(path)   
        print("Carpeta eliminada.")
    
    data_dir = tf.keras.utils.get_file('flower_photos', origin=url, untar=True, cache_dir='.', cache_subdir='')
    return pathlib.Path(data_dir)

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = setup_dataset(dataset_url)

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
    if not np.isclose(train_size + test_size + val_size, 1.0, atol=1e-6):
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
            destination_file = dir_path / file.name
            if not destination_file.exists():
                shutil.copy(file, dir_path)

# Aplicación de la función a cada categoría
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
print("-"*50)

val_dir = pathlib.Path(base_val_dir)
val_images = list(val_dir.glob('*/*.jpg')) #list of all images (full path)
print('\nDirectorio de validación: ', val_dir)
print('Número de imágenes conjunto validación: ', len(val_images))

data_dir = pathlib.Path(base_train_dir)
folder = list(data_dir.glob('*'))
images = list(data_dir.glob('*/*.jpg')) #list of all images (full path)
print('\nDirectorio de entrenamiento: ', data_dir)
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

model_path = './modelo_1000_epocas_64_batchsize_imagenes_iniciales.keras'
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

#Función para predecir una imagen
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (image_size, image_size))
    img = np.reshape(img, [1, image_size, image_size, 3])
    x = img / 255.0
    pred = model.predict(x)
    predicted_class = class_names[np.argmax(pred)]
    percentage = np.max(pred)
    return image_path, predicted_class, round(percentage * 100, 2)  # Devuelve también la ruta de la imagen

# Función para procesar imágenes por categoría
def process_images_by_category(base_dir, num_images=5):
    categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    results = {}
    for category in categories:
        category_path = os.path.join(base_dir, category)
        images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith('.jpg')]
        selected_images = random.sample(images, min(num_images, len(images)))  # Selecciona 5 imágenes o menos si no hay suficientes
        results[category] = [predict_image(img) for img in selected_images]
    return results

# Ruta al directorio de test (para la predicción)
test_dir = './flower_photos/test'

# Ejecuta la predicción
predictions = process_images_by_category(test_dir)
for category, results in predictions.items():
    print(f"Categoría: {category}")
    for image_path, predicted_class, confidence in results:
        print(f"Imagen: {image_path} - Predicha como {predicted_class} con una confianza del {confidence}%")