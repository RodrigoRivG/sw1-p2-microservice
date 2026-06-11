from fastapi import FastAPI
from pydantic import BaseModel
from models.delay_risk_model import predict_delay_risk
from models.best_route_model import predict_best_route
from models.anomaly_model import predict_anomaly

app = FastAPI()

class ProcedureFeatures(BaseModel):
    num_nodes: int
    num_parallel: int
    avg_node_time: float
    department_load: float
    hour_of_day: int
    day_of_week: int

class RouteFeatures(BaseModel):
    route_a_avg_time: float
    route_b_avg_time: float
    route_a_load: float
    route_b_load: float
    route_a_nodes: int
    route_b_nodes: int

@app.get("/")
def root():
    return {"status": "PolicyFlow Deep Learning Service running"}

@app.post("/predict/delay-risk")
def delay_risk(features: ProcedureFeatures):
    return predict_delay_risk(
        features.num_nodes,
        features.num_parallel,
        features.avg_node_time,
        features.department_load,
        features.hour_of_day,
        features.day_of_week
    )

@app.post("/predict/anomaly")
def anomaly(features: ProcedureFeatures):
    return predict_anomaly(
        features.num_nodes,
        features.num_parallel,
        features.avg_node_time,
        features.department_load,
        features.hour_of_day,
        features.day_of_week
    )

@app.post("/predict/best-route")
def best_route(features: RouteFeatures):
    return predict_best_route(
        features.route_a_avg_time,
        features.route_b_avg_time,
        features.route_a_load,
        features.route_b_load,
        features.route_a_nodes,
        features.route_b_nodes
    )

if __name__ == "__main__": 
    import uvicorn 
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)