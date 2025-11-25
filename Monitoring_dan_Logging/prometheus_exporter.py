from flask import Flask, request, jsonify, Response
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')  
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')  

INFERENCE_COUNT = Counter('model_inference_total', 'Total number of inference requests')
INFERENCE_LATENCY = Histogram('model_inference_duration_seconds', 'Inference request latency')
INFERENCE_PREDICTION_DISTRIBUTION = Counter('model_predictions', 'Distribution of predictions', ['prediction'])
INFERENCE_ERRORS = Counter('model_inference_errors', 'Total number of failed inferences')
INFERENCE_LATENCY_BY_PREDICTION = Histogram('model_latency_by_prediction', 'Latency grouped by prediction type', ['prediction'])
INFERENCE_HIGH_RISK_PREDICTIONS = Counter('model_high_risk_predictions_total', 'Total high risk predictions (churn=1)')
INFERENCE_REQUEST_RATE = Counter('model_request_rate_total', 'Total requests for rate calculation')
INFERENCE_FEATURE_DRIFT = Gauge('model_feature_drift', 'Feature drift metrics', ['feature_name', 'metric_type'])

feature_stats = {}

@app.route('/metrics', methods=['GET'])
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=1))  
    RAM_USAGE.set(psutil.virtual_memory().percent)  
    
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/inference', methods=['POST'])
def inference():
    data = request.get_json()
        
    latency = data.get('latency', 0)
    prediction = data.get('prediction', 'unknown')
    success = data.get('success', True)
    confidence = data.get('confidence', None)
    features = data.get('features', {})
    
    INFERENCE_COUNT.inc()
    INFERENCE_LATENCY.observe(latency)
    INFERENCE_REQUEST_RATE.inc()
    
    if success:
        INFERENCE_PREDICTION_DISTRIBUTION.labels(prediction=str(prediction)).inc()
        
        INFERENCE_LATENCY_BY_PREDICTION.labels(prediction=str(prediction)).observe(latency)
        
        if str(prediction) == '1':
            INFERENCE_HIGH_RISK_PREDICTIONS.inc()
        
        if features:
            update_feature_drift(features)
    else:
        INFERENCE_ERRORS.inc()
        
    return jsonify({
        "status": "recorded",
        "latency": latency,
        "prediction": prediction,
        "success": success
    })

def update_feature_drift(features):
    for feature_name, value in features.items():
        if isinstance(value, (int, float)):
            if feature_name not in feature_stats:
                feature_stats[feature_name] = {'sum': 0, 'count': 0, 'sum_sq': 0}
            
            stats = feature_stats[feature_name]
            stats['sum'] += value
            stats['count'] += 1
            stats['sum_sq'] += value ** 2
            
            mean = stats['sum'] / stats['count']
            variance = (stats['sum_sq'] / stats['count']) - (mean ** 2)
            std = variance ** 0.5 if variance > 0 else 0
            
            INFERENCE_FEATURE_DRIFT.labels(feature_name=feature_name, metric_type='mean').set(mean)
            INFERENCE_FEATURE_DRIFT.labels(feature_name=feature_name, metric_type='std').set(std)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)