ngrok http --domain=stable-funny-rooster.ngrok-free.app 5000


# Data Science proyecto

## Requisitos
- Versión de Python: 3.9
- Versión de TensorFlow: 2.10 (permite usar GPU)
- Configuración de TensorFlow con GPU (NVIDIA): Tutorial configurar tensorflow con gpu (NVIDIA): https://youtu.be/C1en3qSs39g?si=x2C_oDaPxDvcUejs
- Requiere instalar Nvidia CUDA(11.2) con Nvidia Cudnn compatible
- Batchsize máximo soportado por la gráfica: 64 (con 128 no funciona)


## Descripción del Script app.py
- El script app.py es una aplicación que utiliza un modelo de clasificación de imágenes previamente entrenado para predecir las clases de flores capturadas en tiempo real por la cámara predeterminada del PC. Muestra los resultados en la pantalla con las tres predicciones más probables y sus respectivas confianzas.

1. Funcionalidades Principales
- Carga del Modelo Preentrenado:
    - Carga el modelo de clasificación de imágenes desde una ruta especificada si existe.

- Captura de Video en Tiempo Real:
    - Utiliza la cámara predeterminada del PC para capturar video en tiempo real.

- Predicción de Imágenes:
    - Redimensiona y normaliza cada cuadro capturado por la cámara.
    - Realiza predicciones utilizando el modelo cargado.
    - Muestra las tres predicciones más probables junto con sus porcentajes de confianza.

- Visualización de Resultados:
    - Muestra las predicciones en la ventana de video en tiempo real.

- Interacción del Usuario:
    - Permite salir de la aplicación presionando la tecla 'q'.

## Descripción del Script datos.py
- El script datos.py se encarga de preparar y organizar los datos para el entrenamiento de modelos de clasificación de imágenes. Este script descarga un conjunto de datos de imágenes, los organiza en carpetas específicas para los conjuntos de entrenamiento, validación y prueba, y distribuye las imágenes de manera aleatoria entre estos conjuntos.

1. Funcionalidades Principales
- Descarga y Configuración del Conjunto de Datos:
    - Descarga el conjunto de datos desde una URL proporcionada.
    - Descomprime el archivo descargado y organiza las imágenes en una estructura de carpetas adecuada.

- Creación de Directorios:
    - Crea los directorios para los conjuntos de datos de entrenamiento (train), validación (val) y prueba (test).

- División de Datos:
    - Divide las imágenes de cada categoría en los conjuntos de entrenamiento, validación y prueba según los porcentajes especificados.

- Copia las imágenes a los directorios correspondientes.

## Descripción del Script entrenar.py
- El script entrenar.py permite entrenar un modelo de clasificación de imágenes utilizando una arquitectura personalizada o ResNet50 preentrenada. El script proporciona flexibilidad para modificar varios hiperparámetros clave como el número de épocas, el tamaño del lote (batch_size) y la paciencia (patience) para el criterio de EarlyStopping.

1. Funcionalidades Principales
- Configuración de Directorios:
    - Define los directorios base para los conjuntos de datos de entrenamiento, validación y prueba.

- Contar Imágenes:
    - Cuenta el número de imágenes en los directorios de entrenamiento y validación.

- Aumento de Datos:
    - Configura un generador de datos con técnicas de aumento como rotación, desplazamiento, corte, zoom, volteo horizontal y vertical.

- Modelos:
    - Modelo Personalizado: Crea un modelo CNN personalizado con capas de convolución, normalización por lotes, agrupación y capas densas.
    - Modelo ResNet50: Configura un modelo ResNet50 preentrenado con capas adicionales de ajuste fino.

- Entrenamiento:
    - Entrena el modelo seleccionado (personalizado o ResNet50) con los parámetros especificados (épocas, batch_size, patience).

- Evaluación y Guardado:
    - Evalúa el rendimiento del modelo en los conjuntos de datos de entrenamiento y validación.
    - Guarda el modelo entrenado, las gráficas de precisión y pérdida, y un archivo de texto con los resultados finales.

2. Parámetros Ajustables
    - Número de Épocas (epochs): Número de iteraciones sobre todo el conjunto de datos de entrenamiento.
    - Tamaño del Lote (batch_size): Número de muestras que se procesan antes de actualizar los parámetros del modelo.
    - Paciencia (patience): Número de épocas sin mejora en la pérdida de validación antes de detener el entrenamiento anticipadamente (early stopping).
