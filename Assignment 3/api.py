from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
from utils import *
from dense_neural_class import *


app = FastAPI(title="HandwrittenDigitClassifier REST API")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


class ImageData(BaseModel):
    image: list


@app.post("/predict")
def predict_digit(image_data: ImageData):
    try:
        image_vector = np.array(image_data.image).reshape(1, -1)
        prediction = model.predict(image_vector)[0]
        return {"predicted_digit": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7000)
