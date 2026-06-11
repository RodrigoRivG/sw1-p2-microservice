FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generar datos y entrenar modelos durante el build
RUN python training/generate_data.py
RUN python models/delay_risk_model.py
RUN python models/anomaly_model.py
RUN python models/best_route_model.py

EXPOSE 8000

CMD ["python", "main.py"]
