Documentación de Pasos para Hacer Accesible una Aplicación Flask
1. Abrir el Puerto 5000 en el Router
Para hacer que una aplicación Flask sea accesible desde el exterior, debes abrir el puerto 5000 en tu router. Aquí están los pasos para hacerlo:

Acceder a la Configuración del Router:

Abre un navegador web y escribe la dirección IP de tu router (por ejemplo, 192.168.1.1 o 192.168.0.1).
Inicia sesión con las credenciales de administrador de tu router.
Configurar el Reenvío de Puertos:

Busca la sección de "Port Forwarding" o "Virtual Server".
Agrega una nueva regla para el puerto 5000.
Description: Api Datascience
Inbound Port: 5000 a 5000
Format: TCP
Private IP Address: La dirección IPv4 local de tu computadora (puedes encontrarla ejecutando ipconfig en Windows o ifconfig en Linux/Mac).
Local Port: 5000 a 5000
Guarda la configuración y reinicia el router si es necesario.
2. Buscar tu IP Pública
Necesitas conocer tu dirección IP pública para acceder a tu aplicación Flask desde el exterior.

Obtener tu Dirección IP Pública:
Abre un navegador web y visita whatismyip.com.
La página mostrará tu dirección IP pública.
3. Configurar y Ejecutar tu Aplicación Flask
Asegúrate de que tu aplicación Flask esté configurada para escuchar en todas las interfaces de red.

Código de Ejemplo para Flask:

python
Copiar código
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
Ejecutar tu Aplicación Flask:

Abre una terminal en tu computadora.
Navega hasta el directorio donde se encuentra tu archivo .py de Flask.
Ejecuta el archivo con el siguiente comando:
bash
Copiar código
python nombre_de_tu_archivo.py
4. Acceder a tu Aplicación desde el Exterior
Acceso Externo:
Desde una computadora externa, abre un navegador web e ingresa la dirección IP pública de tu red seguida del puerto 5000. Ejemplo:
http
Copiar código
http://tu-ip-publica:5000
Alternativa: Usar Ngrok para Hacer tu Aplicación Accesible
Si prefieres no configurar el reenvío de puertos, puedes usar ngrok para hacer que tu aplicación Flask sea accesible públicamente sin exponer tu IP pública.

Instalar Ngrok:

Descarga e instala ngrok desde ngrok.com.
Obtener y Configurar tu Authtoken:

Regístrate en ngrok.
Inicia sesión y copia tu authtoken desde https://dashboard.ngrok.com/get-started/your-authtoken.
Abre una terminal y ejecuta:
bash
Copiar código
ngrok authtoken tu_authtoken
Iniciar un Túnel HTTP:

Abre una terminal y ejecuta:
bash
Copiar código
ngrok http 5000
Ejecutar tu Aplicación Flask:

Abre otra terminal y ejecuta:
bash
Copiar código
python nombre_de_tu_archivo.py
Acceder a tu Aplicación:

Usa la URL pública proporcionada por ngrok para acceder a tu aplicación desde cualquier lugar.
Resumen
Abrir el Puerto 5000 en el Router:

Configura el reenvío de puertos para el puerto 5000 a la IP local de tu computadora.
Buscar tu IP Pública:

Obtén tu dirección IP pública desde whatismyip.com.
Configurar y Ejecutar tu Aplicación Flask:

Asegúrate de que Flask escuche en todas las interfaces (0.0.0.0) y ejecuta tu aplicación.
Acceder a tu Aplicación desde el Exterior:

Usa tu IP pública y el puerto 5000 para acceder a tu aplicación.
Alternativa con Ngrok:

Usa ngrok para crear un túnel seguro sin necesidad de configurar el reenvío de puertos.
Siguiendo estos pasos, podrás hacer que tu aplicación Flask sea accesible desde el exterior.