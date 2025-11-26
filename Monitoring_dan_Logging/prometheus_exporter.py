from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from flask import Flask, request, jsonify, Response
import psutil
import sys

app = Flask(__name__)

CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')  
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')  

INFERENCE_COUNT = Counter('model_inference_total', 'Total inference requests')
INFERENCE_LATENCY = Histogram('model_inference_duration_seconds', 'Inference request latency')
INFERENCE_THROUGHPUT = Counter('model_request_rate_total', 'Total requests per second')
INFERENCE_PREDICTION_DISTRIBUTION = Counter('model_predictions', 'Distribution of predictions', ['prediction'])
INFERENCE_ERRORS = Counter('model_inference_errors', 'Total failed inferences')
INFERENCE_LATENCY_BY_PREDICTION = Histogram('model_latency_by_prediction', 'Latency grouped by prediction type', ['prediction'])
INFERENCE_HIGH_RISK_PREDICTIONS = Counter('model_high_risk_predictions_total', 'Total high risk predictions')
INFERENCE_SIZE_BYTES = Histogram('model_payload_size_bytes', 'Size of payload in bytes', buckets=[100, 500, 1000, 5000, 10000, 50000])

@app.route('/metrics', methods=['GET'])
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=1))  
    RAM_USAGE.set(psutil.virtual_memory().percent)  
    
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/inference', methods=['POST'])
def inference():
    try:
        data = request.get_json()
        latency = data.get('latency', 0)
        prediction = data.get('prediction', 'unknown')
        success = data.get('success', True)
        
        try:
            payload_size = sys.getsizeof(str(data))
            INFERENCE_SIZE_BYTES.observe(payload_size)
        except:
            payload_size = 0
        
        INFERENCE_COUNT.inc()
        INFERENCE_LATENCY.observe(latency)
        INFERENCE_THROUGHPUT.inc()
        
        if success:
            INFERENCE_PREDICTION_DISTRIBUTION.labels(prediction=str(prediction)).inc()
            INFERENCE_LATENCY_BY_PREDICTION.labels(prediction=str(prediction)).observe(latency)
            
            if str(prediction) == '1':
                INFERENCE_HIGH_RISK_PREDICTIONS.inc()
        else:
            INFERENCE_ERRORS.inc()
            
        return jsonify({
            "status": "recorded",
            "latency": latency,
            "prediction": prediction,
            "success": success,
            "payload_size": payload_size
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)