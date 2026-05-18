from fastapi import FastAPI
from pydantic import BaseModel
import joblib


app = FastAPI()


# Load model
model = joblib.load("iris_model.pkl")


# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Home route
@app.get("/")
def home():
    return {
        "message": "Iris FastAPI Running Successfully"
    }


# Prediction route
@app.post("/predict")
def predict(data: IrisInput):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)

    return {
        "predicted_species": prediction[0]
    }