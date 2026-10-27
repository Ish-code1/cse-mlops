from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/Iris_random_forest_model.joblib")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Iris Random Forest API")

#validation of input data using Pydantic
class IrisInput(BaseModel):
	sepal_length: float = Field(gt=0) #greater than 0
	sepal_width: float = Field(gt=0)
	petal_length: float = Field(gt=0)
	petal_width: float = Field(gt=0)

@app.get("/health") ## at url http://localhost:8000/health
def health_check():
	return{"status": "health"}

@app.get("/") # at url http://localhost:8000/
def health_check():
	return{"status": "ready to serve"}

#14/10/26

@app.post("/predict")
def predict(data: IrisInput):
	features =[[
		data.sepal_length,
		data.sepal_width,
		data.petal_length,
		data.petal_width
	]]
	prediction =int(model.predict(features)[0])
	return{
		"prediction" : prediction,
		"class_name" : ["setosa","versicolor","virginica"][prediction]
	}

#api testing/ backend testing - postman , swagger , Thunder client (extention)