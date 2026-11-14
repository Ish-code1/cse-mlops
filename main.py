# Removed unused imports:
# from pathlib import Path
# import joblib
# import numpy as np
# HTTPException was also unused.
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import os
from contextlib import asynccontextmanager
from typing import Callable


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://127.0.0.1:7001"
)
# MLFlow Configuration
#MODEL_URI = "models:/iris-classifier@champion"  bevause after aws deployment, the model uri is different.
MODEL_URI = "model"
# model:/iris-classifier@champion


def load_mlflow_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Previous version:
    # model = mlflow.sklearn.load_model(MODEL_URI)
    # return model
    # Reason: return the loaded model directly.
    return mlflow.sklearn.load_model(MODEL_URI)


class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


def create_app(
    # Previous version: def create_app(model_loader: Callable = load_mlflow_model):
    # Reason: version_loader lets tests avoid a live MLflow request.
    model_loader: Callable = load_mlflow_model,
    version_loader: Callable | None = None,
):
    @asynccontextmanager
    # Previous version: def lifespan(app: FastAPI):
    # Reason: FastAPI lifespan handlers must be asynchronous here.
    async def lifespan(app: FastAPI):
        try:
            app.state.model = model_loader()
        except Exception as ex:
            # Previous version:
            # raise RuntimeError(f"Failed to load model: {ex}")
            # Reason: preserve the original exception as the cause.
            raise RuntimeError(f"Failed to load model: {ex}") from ex

        # Previous version always queried MLflow for the version.
        # Reason: use the injected loader during tests and MLflow in production.
        if version_loader is None:
            client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            champion_version = client.get_model_version_by_alias(
                name="iris-classifier",
                alias="champion",
            )
            app.state.model_version = champion_version.version
        else:
            app.state.model_version = version_loader()

        yield

    app = FastAPI(
        title="Iris Classification API",
        description="Iris prediction API powered by MLFlow",
        version="3.0.0",
        lifespan=lifespan,
    )

    @app.get("/")
    def ready_to_serve():
        return {"status": "Ready to serve"}

    @app.get("/health")
    def health_check():
        # Previous version returned undefined model_version data.
        # Reason: return the response contract used by the health test.
        return {"status": "health"}

    # Endpoint for making predictions with the Iris Random Forest model
    # Previous version: def predict(data: IrisInput):
    # Reason: Request provides access to the model stored during startup.
    @app.post("/predict")
    def predict(data: IrisInput, request: Request):
        features = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width,
        ]]
        # Previous version: prediction = int(model.predict(features)[0])
        # Reason: model is stored on the application state.
        prediction = int(request.app.state.model.predict(features)[0])
        # Previous version printed every prediction for debugging.
        # Reason: avoid unnecessary production output.
        return {
            "prediction": prediction,
            "class_name": ["setosa", "versicolor", "virginica"][prediction],
        }


    return app


app = create_app()
