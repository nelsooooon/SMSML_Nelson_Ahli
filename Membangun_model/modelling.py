import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("http://127.0.0.1:8080/")
mlflow.set_experiment("Auto Logging Random Forest Classifier")
mlflow.autolog(log_models=True)

with mlflow.start_run():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn_preprocessing.csv")
    df.head()
    
    x = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    model_forest = RandomForestClassifier()
    model_forest.fit(X_train, y_train)