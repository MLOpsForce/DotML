from fastapi import FastAPI
import pickle
import os

from pydantic import BaseModel

app = FastAPI()


# load the model
model_path = os.getcwd()+"\\app\\classifier\\model\\"+"rf_model_2022-03-07.sav"
loaded_model = pickle.load(open(model_path, 'rb'))

class HealthDataItem(BaseModel):
    HighBP: str
    HighChol: str
    CholCheck: str
    BMI: str
    Smoker: str
    Stroke: str
    Diabetes: str
    PhysActivity: str
    Fruits: str
    Veggies: str
    HvyAlcoholConsump: str
    AnyHealthcare: str
    NoDocbcCost: str
    GenHlth: str
    MentHlth: str
    PhysHlth: str
    DiffWalk: str
    Sex: str
    Age: str
    Education: str
    Income: str


@app.get("/prediction")
async def prediction(row: HealthDataItem):
    
    print("row:", row)

    # transform data to match requirements if needed

    # call model prediction and return it, maybe with a certainty

    
    
    return {"message": "Hello World"}