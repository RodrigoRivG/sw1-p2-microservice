import numpy as np
import pandas as pd

def generate_procedure_data(n_samples=1000):
    np.random.seed(42)

    # Simular datos de trámites
    data = {
        'num_nodes': np.random.randint(2, 10, n_samples),
        'num_parallel': np.random.randint(0, 3, n_samples),
        'avg_node_time': np.random.uniform(10, 300, n_samples),
        'department_load': np.random.uniform(0.1, 1.0, n_samples),
        'hour_of_day': np.random.randint(7, 18, n_samples),
        'day_of_week': np.random.randint(0, 5, n_samples),
    }

    # Target: tiempo total en minutos (con algo de ruido)
    data['total_time'] = (
        data['num_nodes'] * data['avg_node_time'] +
        data['num_parallel'] * 50 +
        data['department_load'] * 100 +
        np.random.normal(0, 20, n_samples)
    )

    # Target: riesgo de demora (1 = demora, 0 = normal)
    data['delay_risk'] = (data['total_time'] > 400).astype(int)

    # Target: anomalía (1 = anómalo, 0 = normal)
    data['is_anomaly'] = (
        (data['avg_node_time'] > 250) | 
        (data['department_load'] > 0.9)
    ).astype(int)

    return pd.DataFrame(data)

def generate_route_data(n_samples=1000):
    np.random.seed(42)

    data = {
        'route_a_avg_time': np.random.uniform(10, 200, n_samples),
        'route_b_avg_time': np.random.uniform(10, 200, n_samples),
        'route_a_load': np.random.uniform(0.1, 1.0, n_samples),
        'route_b_load': np.random.uniform(0.1, 1.0, n_samples),
        'route_a_nodes': np.random.randint(1, 5, n_samples),
        'route_b_nodes': np.random.randint(1, 5, n_samples),
    }

    # Target: 0 = ruta A es mejor, 1 = ruta B es mejor
    data['best_route'] = (
        (data['route_b_avg_time'] * data['route_b_load']) -
        (data['route_a_avg_time'] * data['route_a_load'])
    ).astype(int)

    return pd.DataFrame(data)

if __name__ == "__main__":
    df_procedure = generate_procedure_data()
    df_route = generate_route_data()
    
    df_procedure.to_csv('training/procedure_data.csv', index=False)
    df_route.to_csv('training/route_data.csv', index=False)
    
    print(f"Datos de trámites generados: {len(df_procedure)} muestras")
    print(f"Datos de rutas generados: {len(df_route)} muestras")
    print(df_procedure.head())