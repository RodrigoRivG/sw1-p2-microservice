import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os

def train_delay_risk_model():
    # Cargar datos
    df = pd.read_csv('training/procedure_data.csv')
    
    features = ['num_nodes', 'num_parallel', 'avg_node_time', 
                 'department_load', 'hour_of_day', 'day_of_week']
    
    X = df[features].values
    y = df['delay_risk'].values
    
    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Modelo
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
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
    
    # Guardar modelo y scaler
    os.makedirs('trained', exist_ok=True)
    model.save('trained/delay_risk_model.keras')
    with open('trained/delay_risk_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Modelo de riesgo de demora guardado")

def predict_delay_risk(num_nodes, num_parallel, avg_node_time, 
                        department_load, hour_of_day, day_of_week):
    model = tf.keras.models.load_model('trained/delay_risk_model.keras')
    with open('trained/delay_risk_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    features = np.array([[num_nodes, num_parallel, avg_node_time,
                          department_load, hour_of_day, day_of_week]])
    features_scaled = scaler.transform(features)
    
    risk = float(model.predict(features_scaled)[0][0])
    return {
        "delay_risk": risk,
        "risk_level": "alto" if risk > 0.7 else "medio" if risk > 0.4 else "bajo"
    }

if __name__ == "__main__":
    train_delay_risk_model()