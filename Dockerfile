# 1. Imagen Base de Python
FROM python:3.9-slim

# 2. Instalar Dependencias del Sistema (Tesseract y Librerías de OpenCV)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libgthread-2.0-0

# 3. Directorio de Trabajo
WORKDIR /app

# 4. Copiar Archivo de Dependencias
COPY requirements.txt .

# 5. Instalar Dependencias de Python 
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copiar el Código de la Aplicación
COPY . .

# 7. Exponer Puerto para Render
EXPOSE 10000

# 8. Comando de Inicio para Flask con Gunicorn
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:10000", "server:app"]

