#Import TestClient so we can send simulated HTTP requests directly to FASTAPI

from fastapi import testclient
from fastapi.testclient import TestClient 
#import the application object defined in main.py
# from main import app
from main import create_app
import pytest
#create a reusable test client for all test function
# client = TestClient(app)

class FakeIrisModel:
    def predict(self, features):
        #return a fixed prediction for testing purposes
        return [0]  # always predict "setosa"
def fake_model_loader():
    return FakeIrisModel()
def fake_version_loader():
    return "test-version"

@pytest.fixture
def client():
    #create a test client with the fake model loader
    app = create_app(
        model_loader=fake_model_loader,
        version_loader=fake_version_loader
        )
    with TestClient(app) as test_client:
        yield test_client

def test_health_endpoint(client):
    """
    Verify that the application is running and can respond to a health-check request
    """
    response = client.get("/health")
    assert response.status_code == 200 # a successfull http request shd return status code 200

    #now check rsponse body
    assert response.json() == {"status": "health"}

def test_valid_iris_prediction(client):
    """
    Verify that valid flower measurements produce a prediction
    """
    payload = {
        "sepal_length": 3.4,
        "sepal_width": 1.4,
        "petal_length":3.4,
        "petal_width":0.4
    }
    #send a simulated http post request with json input
    response = client.post("/predict",json=payload)
    assert response.status_code ==200
    #convert the json response into python dictionary
    result = response.json()

    #check the response contract: both fields must exist
    assert "prediction" in result
    assert "class_name" in result

    #check that the prediction is a valid class index and class name is a string
    assert result["class_name"] in ["setosa", "versicolor", "virginica"]
    assert result["prediction"] in [0,1,2]

def test_invalid_iris_input(client):
     payload = {
            "sepal_length": -3.4,
            "sepal_width": 1.4,
            "petal_length":3.4,
            "petal_width":0.4
        }
        #send a simulated http post request with json input
     response = client.post("/predict",json=payload)
    #fastapi automatically returns a HTTP 422 status code for invalid input data
     assert response.status_code == 422
