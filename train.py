from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

iris = load_iris()
#(use GitHub copilot now for the same repetitive code of training till accuracy score. Main is of fastapi to be written manually)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

y_pred = clf.predict(X_test)
print(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}")

#save the trained model 
model_dir = Path("models")
model_dir.mkdir(parents=True,exist_ok=True)
model_path = model_dir/"Iris_random_forest_model.joblib" #MODEL SERIALIZATION
joblib.dump(clf,model_path)
print(f"Model saved to {model_path}")
