import mlflow
import dagshub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score, confusion_matrix, classification_report

np.random.seed(40)

warnings.filterwarnings("ignore")

dagshub.init(repo_owner='nelsooooon', repo_name='SMSML_Nelson_Ahli', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/nelsooooon/SMSML_Nelson_Ahli.mlflow")
mlflow.set_experiment("Logging Model")

df = pd.read_csv("Membangun_model/WA_Fn-UseC_-Telco-Customer-Churn_preprocessing.csv")

x = df.drop(columns=['Churn'])
y = df['Churn']

input_example = x.head(5)
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name=f"elastic_search_{n_estimators}_{max_depth}"):
    model_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    
    param_dist = {
    'n_estimators': np.linspace(100, 500, 5, dtype=int),
    'max_depth': np.linspace(10, 50, 5, dtype=int),
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
    }

    random_search = RandomizedSearchCV(estimator=model_forest, param_distributions=param_dist, n_iter=20, cv=3, n_jobs=-1, verbose=2, random_state=42)
    random_search.fit(X_train, y_train)
        
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        input_example=input_example
    )
    
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    log_loss_score = log_loss(y_test, y_prob)
    roc = roc_auc_score(y_test, y_prob[:, 1])
    score = best_model.score(X_test, y_test)
    
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Random Search Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    plt.savefig('Membangun_model/artifacts/confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    feature_importance = pd.DataFrame({
        'feature': x.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Random Forest Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plt.savefig('Membangun_model/artifacts/feature_importance.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    report = classification_report(y_test, y_pred)
    with open('Membangun_model/artifacts/classification_report.txt', 'w') as f:
        f.write('Classification Report\n')
        f.write('='*50 + '\n\n')
        f.write(report)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("log_loss", log_loss_score)
    mlflow.log_metric("roc_auc", roc)
    mlflow.log_metric("score", score)
    
    mlflow.log_artifact('Membangun_model/artifacts/confusion_matrix.png')
    mlflow.log_artifact('Membangun_model/artifacts/feature_importance.png')
    mlflow.log_artifact('Membangun_model/artifacts/classification_report.txt')
    
    mlflow.log_params(best_params)