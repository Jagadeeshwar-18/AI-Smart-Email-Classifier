from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.predict import predict_email

app = FastAPI(title="AI Smart Email Classifier API")

class EmailRequest(BaseModel):
    email: str

@app.post("/predict")
def predict(req: EmailRequest):
    return predict_email(req.email)
