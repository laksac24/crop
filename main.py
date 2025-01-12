import os
import gzip
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class ModelInput(BaseModel):
    N: int
    P: int
    K: int
    temperature: float
    humidity: float
    ph: float
    rainfall: float


with gzip.open('model1.pkl.gz', 'rb') as f:
    model = pickle.load(f)

@app.post('/predict')
def prediction(input_param: ModelInput):
    # Convert input to dictionary
    input_data = input_param.dict()

    nitrogen = input_data['N']
    phosphorous = input_data['P']
    potassium = input_data['K']
    temp = input_data['temperature']
    humid = input_data['humidity']
    phv = input_data['ph']
    rain = input_data['rainfall']

    input_list = [nitrogen, phosphorous, potassium, temp, humid, phv, rain]

    prediction = model.predict([input_list])

    crop_map = {
        1: 'apple',
        2: 'banana',
        3: 'rice',
        4: 'pomegranate',
        5: 'pigeonpeas',
        6: 'papaya',
        7: 'orange',
        8: 'muskmelon',
        9: 'mungbean',
        10: 'mothbeans',
        11: 'mango',
        12: 'maize',
        13: 'lentil',
        14: 'kidneybeans',
        15: 'jute',
        16: 'grapes',
        17: 'cotton',
        18: 'coffee',
        19: 'coconut',
        20: 'chickpea',
        21: 'blackgram',
        22: 'watermelon'
    }

    
    predicted_crop = crop_map.get(prediction[0], "Unknown crop")

    return JSONResponse(content={"predicted_crop": predicted_crop})

@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Recommendation API"}
