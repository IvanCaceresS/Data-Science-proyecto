import os
from pathlib import Path
import time
import PIL
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, InputLayer, Conv2D, MaxPooling2D, BatchNormalization, GlobalMaxPooling2D, Dropout
from tensorflow.keras.models import Model

# Definir los directorios base
base_dir = Path('./flower_photos')  # Directorio con las categorías originales
base_train_dir = Path('./flower_photos/train')
base_test_dir = Path('./flower_photos/test')
base_val_dir = Path('./flower_photos/val')

val_dir = Path(base_val_dir)
val_images = list(val_dir.glob('*/*.jpg'))  # list of all images (full path)
print('\nCarpeta de validación: ', val_dir)
print('Número de imágenes conjunto validación: ', len(val_images))

data_dir = Path(base_train_dir)
folder = list(data_dir.glob('*'))
images = list(data_dir.glob('*/*.jpg'))  # list of all images (full path)
print('\nCarpeta de entrenamiento: ', data_dir)
print('Número de imágenes conjunto entrenamiento: ', len(images))
print("-" * 80, '\n')

image_size = 256
batch_size = 32

idg = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,       # Rotar las imágenes en un rango de hasta 40 grados
    width_shift_range=0.2,   # Desplazamientos horizontales
    height_shift_range=0.2,  # Desplazamientos verticales
    shear_range=0.2,         # Corte de imágenes
    zoom_range=0.2,          # Zoom aleatorio
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'      # Rellenar píxeles que podrían quedar vacíos después de una transformación
)

train_gen = idg.flow_from_directory(
    base_train_dir,
    target_size=(image_size, image_size),
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    seed=1
)

val_gen = idg.flow_from_directory(
    base_val_dir,
    target_size=(image_size, image_size),
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    seed=1
)

# Classes train_gen
classes = train_gen.class_indices
print(classes)
class_names = list(classes.keys())
print('Los nombres de las clases de train son: ', class_names)

print('\n', "-" * 80, '\n')

# Classes val_gen
classes = val_gen.class_indices
print(classes)
class_names = list(classes.keys())
print('Los nombres de las clases de val son: ', class_names)

print('\n', "-" * 80, '\n')

# Número de imágenes de entrenamiento
print("Número de batches de entrenamiento: ", len(train_gen))
# Número de imágenes de validación
print("Número de batches de validación: ", len(val_gen))

print('\n', "-" * 80, '\n')

model_path = './modelos/modelo_resnet50_100epocas_32_batchsize.keras'

# Función para configurar ResNet50
def create_resnet_model(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if os.path.exists(model_path):
    print("Cargando modelo existente...: ", model_path)
    model = tf.keras.models.load_model(model_path)
    print("Modelo cargado.")
else:
    print("Entrenando nuevo modelo...")
    # Decidir si usar ResNet50 o el modelo personalizado
    use_resnet = input("¿Deseas usar ResNet50? (s/n): ").lower() == 's' 
    if use_resnet:
        print("Configurando ResNet50...")
        model = create_resnet_model(input_shape=(image_size, image_size, 3), num_classes=len(train_gen.class_indices))
    else:
        print("Configurando el modelo personalizado...")
        model = tf.keras.models.Sequential([
            InputLayer(input_shape=(image_size, image_size, 3)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            GlobalMaxPooling2D(),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(len(class_names), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        train_gen,
        epochs=100,
        steps_per_epoch=len(train_gen.filenames) // batch_size,
        validation_data=val_gen,
        validation_steps=len(val_gen.filenames) // batch_size,
        verbose=1
    )
    model.save(model_path)

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