# import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import load_model  # , inference


# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(...,
                                example="United-States",
                                alias="native-country")


path = "model/encoder.pkl"  # TODO: enter the path for the saved encoder
encoder = load_model(path)

path = "model/model.pkl"  # TODO: enter the path for the saved model
model = load_model(path)

# TODO: create a RESTful API using FastAPI
app = FastAPI()

# TODO: create a GET on the root giving a welcome message


@app.get("/")
async def get_root():
    """Say hello!"""
    # your code here
    return {"message": "Welcome to the model inference API!"}


# TODO: create a POST on a different path that does model inference


@app.post("/predict")
async def post_inference(data: Data):
    data_dict = data.dict()
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    data_processed, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
    )

    _inference = model.predict(data_processed)
    return {"prediction": apply_label(_inference)}
