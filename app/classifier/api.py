from fastapi import FastAPI
import pickle
import os

app = FastAPI()


# load the model
model_path = os.getcwd()+"\\app\\classifier\\model\\"+"rf_model_2022-03-07.sav"
loaded_model = pickle.load(open(model_path, 'rb'))

@app.get("/")
async def prediction():
    # assure it is a get request

    # get data from json

    # transform data to match requirements if needed

    # call model prediction and return it, maybe with a certainty

    
    
    return {"message": "Hello World"}