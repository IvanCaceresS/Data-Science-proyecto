import os
from pathlib import Path
import shutil
import PIL

import numpy as np
import tensorflow as tf

def setup_dataset(url, path='./flower_photos'):
    if os.path.exists(path):
        print("Eliminando la carpeta flower_photos")
        shutil.rmtree(path)   
        print("Carpeta eliminada.")
    
    data_dir = tf.keras.utils.get_file('flower_photos', origin=url, untar=True, cache_dir='.', cache_subdir='')
    return Path(data_dir)

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = setup_dataset(dataset_url)

print("-"*50)

# Crear los directorios para los conjuntos de datos
base_train_dir = data_dir / 'train'
base_test_dir = data_dir / 'test'
base_val_dir = data_dir / 'val'

# Crear directorios de base si no existen
for dir in (base_train_dir, base_test_dir, base_val_dir):
    if not dir.exists():
        os.makedirs(dir)

# Función para dividir los datos en conjuntos de entrenamiento, prueba y validación
def split_data(source, train_dir, test_dir, val_dir, train_size=0.8, test_size=0.10, val_size=0.10):
    files = [file for file in source.iterdir() if file.is_file() and not file.name.startswith('.')]
    print(f"Procesando {source.stem}: {len(files)} archivos encontrados.")  # Diagnóstico
    np.random.shuffle(files)
    files = np.array(files)
    
    train_end = int(train_size * len(files))
    test_end = int((train_size + test_size) * len(files))
    
    train_files = files[:train_end]
    test_files = files[train_end:test_end]
    val_files = files[test_end:]
    
    def copy_files(files, directory):
        os.makedirs(directory, exist_ok=True)
        for file in files:
            destination_file = directory / file.name
            if not destination_file.exists():
                shutil.copy(file, directory)

    copy_files(train_files, train_dir)
    copy_files(test_files, test_dir)
    copy_files(val_files, val_dir)

    # Imprimir la cantidad de archivos copiados por categoría para diagnóstico
    print(f"{source.stem}: {len(train_files)} entrenamiento, {len(test_files)} prueba, {len(val_files)} validación")

# Definir los directorios base
base_dir = Path('./flower_photos')  # Directorio con las categorías originales
base_train_dir = Path('./flower_photos/train')
base_test_dir = Path('./flower_photos/test')
base_val_dir = Path('./flower_photos/val')

for category in base_dir.iterdir():
    if category.is_dir() and category.name not in ['train', 'test', 'val']:
        category_name = category.name
        cat_train_dir = base_train_dir / category_name
        cat_test_dir = base_test_dir / category_name
        cat_val_dir = base_val_dir / category_name
        
        split_data(category, cat_train_dir, cat_test_dir, cat_val_dir)

print("Datos divididos y almacenados correctamente.")