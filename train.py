from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn

#configure MLFlow tracking
mlflow.set_tracking_uri("http://127.0.0.1:7001")
#group related training runs under one experiment
mlflow.set_experiment("iris-classification")


iris = load_iris()
#(use GitHub copilot now for the same repetitive code of training till accuracy score. Main is of fastapi to be written manually)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

n_estimators =10
random_State =42
model_parameters ={
    "model_type":"RandomForestClassifier",
    "n_Estimators":n_estimators,
    "random_state": random_State,
    "test_size":0.2
}
#start a tracked training run
with mlflow.start_run() as run:
    #record a training config before model fitting
    mlflow.log_params(model_parameters)
    #build classifier using selected  parameters

    clf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    

    y_pred = clf.predict(X_test)
    print(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    #calculate the precision
    precision = precision_score(y_test, y_pred, average='macro',zero_division=0)
    print(f"Precision: {precision}")
    #calculate the recall  
    recall = recall_score(y_test, y_pred, average='macro',zero_division=0)
    print(f"Recall: {recall}")

    #Mosdel quality gate
    #define min acceptable accuracy threshold
    MIN_TEST_ACCURACY = 0.9
    if accuracy < MIN_TEST_ACCURACY:
        raise ValueError(
            f"Model rejected:test accuracy:{accuracy:.2f%}"
            f"is below the required {MIN_TEST_ACCURACY}"
        )
    print("Model Quality gate passed...")
    
    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall
    })


    #save the trained model 
    model_dir = Path("models")
    model_dir.mkdir(parents=True,exist_ok=True)
    model_path = model_dir/"Iris_random_forest_model.joblib" #MODEL SERIALIZATION
    joblib.dump(clf,model_path)
    print(f"Model saved to {model_path}")

    #model registration
    mlflow.sklearn.log_model(
        sk_model=clf,
        name= "iris_model",
        registered_model_name="iris-classifier",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
        )
    print("Registered model to MLFlow model registry")