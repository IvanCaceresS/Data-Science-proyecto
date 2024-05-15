import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalMaxPooling2D, Dropout, InputLayer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2

# Definir los directorios base
base_dir = Path('./flower_photos')
train_dir = base_dir / 'train'
test_dir = base_dir / 'test'
val_dir = base_dir / 'val'

# Contar imágenes en los directorios
val_images = list(val_dir.glob('*/*.jpg'))
train_images = list(train_dir.glob('*/*.jpg'))
print(f'\nCarpeta de validación: {val_dir}')
print(f'Número de imágenes conjunto validación: {len(val_images)}')
print(f'\nCarpeta de entrenamiento: {train_dir}')
print(f'Número de imágenes conjunto entrenamiento: {len(train_images)}')
print("-" * 80, '\n')

image_size = 256
batch_size = 32

# Configuración del generador de datos con aumento de datos
data_gen_args = dict(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

idg = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

train_gen = idg.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    seed=1
)

val_gen = idg.flow_from_directory(
    val_dir,
    target_size=(image_size, image_size),
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    seed=1
)

# Imprimir las clases
train_classes = train_gen.class_indices
val_classes = val_gen.class_indices
class_names = list(train_classes.keys())
print(f'Los nombres de las clases de train son: {class_names}')
print(f'Los nombres de las clases de val son: {class_names}')
print("-" * 80, '\n')

# Número de imágenes de entrenamiento y validación
print(f"Número de batches de entrenamiento: {len(train_gen)}")
print(f"Número de batches de validación: {len(val_gen)}")
print("-" * 80, '\n')

model_path = './modelos/modelo_resnet50_1000epocas_32batchsize_200patience.keras'

# Función para configurar ResNet50
def create_resnet_model(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)  # Incrementar dropout
    predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Crear modelo personalizado con regularización
def create_custom_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        InputLayer(input_shape=input_shape),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling2D(2, 2),
        GlobalMaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),  # Incrementar dropout
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Función para graficar métricas y análisis de ajuste
def plot_metrics(history, model_path):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(14, 7))
    
    # Gráfica de precisión
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Precisión Entrenamiento')
    plt.plot(epochs_range, val_acc, label='Precisión Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend(loc='lower right')
    plt.title('Precisión de Entrenamiento y Validación')
    
    # Gráfica de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Pérdida Entrenamiento')
    plt.plot(epochs_range, val_loss, label='Pérdida Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend(loc='upper right')
    plt.title('Pérdida de Entrenamiento y Validación')
    
    # Análisis de overfitting y underfitting
    final_train_acc = acc[-1]
    final_val_acc = val_acc[-1]
    final_train_loss = loss[-1]
    final_val_loss = val_loss[-1]

    overfitting_text = ''
    if (final_train_acc - final_val_acc > 0.1) and (final_val_loss > final_train_loss):
        overfitting_text = "Overfitting detectado"
    elif (final_train_acc < 0.6) and (final_val_acc < 0.6):
        overfitting_text = "Underfitting detectado"
    else:
        overfitting_text = "Ajuste adecuado"

    plt.figtext(0.5, 0.01, f'Análisis: {overfitting_text}', ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    
    # Guardar la gráfica
    plot_path = model_path.replace('.keras', '.jpg')
    plt.savefig(plot_path)
    plt.show()
    
    # Guardar los resultados finales en un archivo de texto
    results_path = model_path.replace('.keras', '.txt')
    stopped_epoch = len(acc)
    total_epochs = history.params['epochs']
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f'Precisión final de Entrenamiento: {final_train_acc:.4f}\n')
        f.write(f'Precisión final de Validación: {final_val_acc:.4f}\n')
        f.write(f'Pérdida final de Entrenamiento: {final_train_loss:.4f}\n')
        f.write(f'Pérdida final de Validación: {final_val_loss:.4f}\n')
        f.write(f'Análisis: {overfitting_text}\n')
        f.write(f'Epoca detenida: {stopped_epoch}/{total_epochs}\n')

if os.path.exists(model_path):
    print(f"Modelo: {model_path} ya existe.")
else:
    print(f"Entrenando nuevo modelo... {model_path}")
    # Decidir si usar ResNet50 o el modelo personalizado
    use_resnet = input("¿Deseas usar ResNet50? (s/n): ").lower() == 's' 
    if use_resnet:
        print("Configurando ResNet50...")
        model = create_resnet_model(input_shape=(image_size, image_size, 3), num_classes=len(train_gen.class_indices))
    else:
        print("Configurando el modelo personalizado...")
        model = create_custom_model(input_shape=(image_size, image_size, 3), num_classes=len(train_gen.class_indices))

    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)

    history = model.fit(
        train_gen,
        epochs=1000,
        steps_per_epoch=len(train_gen),
        validation_data=val_gen,
        validation_steps=len(val_gen),
        verbose=1,
        callbacks=[early_stopping]  # Añade early stopping
    )
    model.save(model_path)
    plot_metrics(history, model_path)
