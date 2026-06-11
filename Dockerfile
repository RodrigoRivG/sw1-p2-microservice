FROM python:3.10-slim

WORKDIR /app

# Copiar requirements.txt e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del microservicio y los modelos entrenados
COPY . .

# Exponer el puerto 8000 en el que corre FastAPI
EXPOSE 8000

# Comando para iniciar el microservicio
CMD ["python", "main.py"]
