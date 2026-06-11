import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os

def train_best_route_model():
    # Cargar datos
    df = pd.read_csv('training/route_data.csv')
    
    features = ['route_a_avg_time', 'route_b_avg_time',
                 'route_a_load', 'route_b_load',
                 'route_a_nodes', 'route_b_nodes']
    
    X = df[features].values
    y = df['best_route'].values
    
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
    model.save('trained/best_route_model.keras')
    with open('trained/best_route_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Modelo de mejor ruta guardado")

def predict_best_route(route_a_avg_time, route_b_avg_time,
                        route_a_load, route_b_load,
                        route_a_nodes, route_b_nodes):
    model = tf.keras.models.load_model('trained/best_route_model.keras')
    with open('trained/best_route_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    features = np.array([[route_a_avg_time, route_b_avg_time,
                          route_a_load, route_b_load,
                          route_a_nodes, route_b_nodes]])
    features_scaled = scaler.transform(features)
    
    prediction = float(model.predict(features_scaled)[0][0])
    best = "B" if prediction > 0.5 else "A"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return {
        "best_route": best,
        "confidence": confidence,
        "explanation": f"La ruta {best} tiene menor carga y tiempo estimado"
    }

if __name__ == "__main__":
    train_best_route_model()