import requests
import time
import pandas as pd

API_URL = "http://127.0.0.1:5005/invocations"
INFERENCE_URL = "http://127.0.0.1:8000/inference"

df = pd.read_csv("Membangun_model/WA_Fn-UseC_-Telco-Customer-Churn_preprocessing.csv")
if 'Churn' in df.columns:
    df = df.drop(columns=['Churn'])

def record_metrics(latency, prediction, features=None, success=True):
    try:
        payload = {
            "latency": latency,
            "prediction": str(prediction),
            "success": success
        }
        
        if features is not None:
            payload["features"] = features
        
        requests.post(INFERENCE_URL, json=payload, timeout=1)
    except:
        pass  
    
def load_sample_data():
    sample = df.sample(n=1).iloc[0]
    return sample.to_dict()

def make_inference(data):
    start_time = time.time()
    
    payload = {"instances": [data]}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=5)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            predictions = result.get('predictions', [])
            prediction = predictions[0] if predictions else 'unknown'
            
            record_metrics(
                latency=response_time,
                prediction=prediction,
                features=data,
                success=True
            )
            
            print(f"✓ Success - Latency: {response_time:.3f}s - Prediction: {prediction}")
            return result
        else:
            record_metrics(response_time, 'error', success=False)
            print(f"✗ Failed - Status: {response.status_code}")
            return None
        
    except Exception as e:
        response_time = time.time() - start_time
        record_metrics(response_time, 'error', success=False)
        print(f"✗ Error: {e}")
        return None

def run_continuous_inference(interval=5, num_requests=100):
    for i in range(num_requests):
        data = load_sample_data()
        
        if data:
            make_inference(data)

        time.sleep(interval)

if __name__ == '__main__':
    run_continuous_inference()