import mlflow
import pandas as pd
import sys
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("http://127.0.0.1:8080/")
mlflow.set_experiment("Logging Model")

df = pd.read_csv("Membangun_model/WA_Fn-UseC_-Telco-Customer-Churn_preprocessing.csv")
df.head()

x = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37

with mlflow.start_run(run_name=f"elastic_search_{n_estimators}_{max_depth}"):
        mlflow.autolog()

        model_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model_forest.fit(X_train, y_train)
        
        accuracy = model_forest.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)