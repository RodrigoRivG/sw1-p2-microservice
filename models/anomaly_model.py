import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os

def train_anomaly_model():
    # Cargar datos
    df = pd.read_csv('training/procedure_data.csv')
    
    features = ['num_nodes', 'num_parallel', 'avg_node_time',
                 'department_load', 'hour_of_day', 'day_of_week']
    
    X = df[features].values
    y = df['is_anomaly'].values
    
    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Modelo
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(X_train, y_train, epochs=20, batch_size=32,
              validation_data=(X_test, y_test), verbose=1)
    
    # Evaluar
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Guardar
    os.makedirs('trained', exist_ok=True)
    model.save('trained/anomaly_model.keras')
    with open('trained/anomaly_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Modelo de detección de anomalías guardado")

def predict_anomaly(num_nodes, num_parallel, avg_node_time,
                    department_load, hour_of_day, day_of_week):
    model = tf.keras.models.load_model('trained/anomaly_model.keras')
    with open('trained/anomaly_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    features = np.array([[num_nodes, num_parallel, avg_node_time,
                          department_load, hour_of_day, day_of_week]])
    features_scaled = scaler.transform(features)
    
    anomaly_score = float(model.predict(features_scaled)[0][0])
    return {
        "anomaly_score": anomaly_score,
        "is_anomaly": anomaly_score > 0.5,
        "severity": "alta" if anomaly_score > 0.8 else "media" if anomaly_score > 0.5 else "normal"
    }

if __name__ == "__main__":
    train_anomaly_model()