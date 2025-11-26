# SMSML_Nelson_Ahli

A Machine Learning project for **Telco Customer Churn Prediction** using Random Forest Classifier, with MLOps practices including experiment tracking (MLflow/DagsHub), model monitoring (Prometheus), and alerting (Grafana).

## Project Overview

This repository contains a complete ML pipeline for predicting customer churn in the telecommunications industry. The project demonstrates:

- **Model Building**: Random Forest classification with hyperparameter tuning
- **Experiment Tracking**: MLflow integration with DagsHub for centralized experiment management
- **Model Monitoring**: Prometheus-based metrics collection for inference monitoring
- **Alerting & Visualization**: Grafana dashboards with alerting rules

## Repository Structure

```
SMSML_Nelson_Ahli/
├── Membangun_model/                    # Model building and training
│   ├── modelling.py                    # Basic model training script
│   ├── modelling_tuning.py             # Model training with hyperparameter tuning
│   ├── requirements.txt                # Python dependencies
│   ├── WA_Fn-UseC_-Telco-Customer-Churn_preprocessing.csv  # Preprocessed dataset
│   ├── artifacts/                      # Model artifacts
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   └── classification_report.txt
│   ├── screenshoot_artifak.png         # Screenshot of MLflow artifacts
│   └── screenshoot_dashboard.png       # Screenshot of MLflow dashboard
│
├── Monitoring_dan_Logging/             # Monitoring and logging infrastructure
│   ├── inference.py                    # Inference client with metrics recording
│   ├── prometheus_exporter.py          # Prometheus metrics exporter (Flask API)
│   ├── prometheus.yml                  # Prometheus configuration
│   ├── bukti_monitoring_Prometheus/    # Prometheus monitoring screenshots
│   ├── bukti_monitoring_Grafana/       # Grafana dashboard screenshots
│   ├── bukti_alerting_Grafana/         # Grafana alerting screenshots
│   └── bukti_serving.png               # Model serving screenshot
│
├── mlruns/                             # Local MLflow experiment data
├── mlartifacts/                        # Local MLflow artifacts
├── Eksperimen_SML_Nelson_Ahli.txt      # Link to experiment repository
└── Workflow-CI.txt                     # Link to CI/CD workflow repository
```

## Features

### Model Building (`Membangun_model/`)

- **Algorithm**: Random Forest Classifier
- **Hyperparameter Tuning**: RandomizedSearchCV with cross-validation
- **Experiment Tracking**: MLflow with DagsHub integration
- **Metrics Logged**:
  - Accuracy, Precision, Recall, F1-Score
  - Log Loss, ROC-AUC, Model Score
- **Artifacts Generated**:
  - Confusion Matrix visualization
  - Feature Importance chart
  - Classification Report

### Monitoring & Logging (`Monitoring_dan_Logging/`)

**Prometheus Metrics**:
- `system_cpu_usage` - CPU usage percentage
- `system_ram_usage` - RAM usage percentage
- `model_inference_total` - Total inference requests
- `model_inference_duration_seconds` - Inference latency histogram
- `model_predictions` - Prediction distribution counter
- `model_inference_errors` - Failed inference counter
- `model_high_risk_predictions_total` - High churn risk predictions
- `model_payload_size_bytes` - Request payload size histogram

**Grafana Dashboards**:
- System resource monitoring (CPU, RAM)
- Inference latency and throughput
- Prediction distribution
- High-risk churn alerts

**Alerting Rules**:
- Error rate alerts
- Latency threshold alerts
- Request rate alerts

## Requirements

Install dependencies:

```bash
pip install -r Membangun_model/requirements.txt
```

### Dependencies

- dagshub==0.6.3
- matplotlib==3.10.7
- mlflow==3.6.0
- numpy==2.3.5
- pandas==2.3.3
- protobuf==6.33.1
- scikit_learn==1.7.2
- seaborn==0.13.2

Additional dependencies for monitoring:
- prometheus_client>=0.19.0
- flask>=3.0.0
- psutil>=5.9.0
- requests>=2.31.0

## Usage

### Model Training

Run basic model training:
```bash
python Membangun_model/modelling.py [n_estimators] [max_depth]
```

Run model training with hyperparameter tuning:
```bash
python Membangun_model/modelling_tuning.py [n_estimators] [max_depth]
```

> **Note:** Both `n_estimators` and `max_depth` are optional arguments.  
> Default values: `n_estimators=505`, `max_depth=37`.

### Start Monitoring Stack

1. Start the Prometheus exporter:
```bash
python Monitoring_dan_Logging/prometheus_exporter.py
```

2. Start Prometheus (requires Prometheus installed):
```bash
prometheus --config.file=Monitoring_dan_Logging/prometheus.yml
```

3. Run inference client:
```bash
python Monitoring_dan_Logging/inference.py
```

## Related Repositories

- **Experiments Repository**: [Eksperimen_SML_Nelson-Ahli](https://github.com/nelsooooon/Eksperimen_SML_Nelson-Ahli)
- **CI/CD Workflow Repository**: [Workflow-CI](https://github.com/nelsooooon/Workflow-CI)
- **DagsHub Tracking**: [SMSML_Nelson_Ahli on DagsHub](https://dagshub.com/nelsooooon/SMSML_Nelson_Ahli/)

## Author

Nelson Ahli

## License

This project is for educational purposes as part of the SML (Sistem Machine Learning) coursework.
