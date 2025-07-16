
# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os


# Define the data schema

class Customer_Data(BaseModel):
    
    CreditScore: int  
    Gender: int   # 0: Female, 1: Male
    Age: int
    Tenure: int # how many years
    Balance: float 
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember :int
    EstimatedSalary: float
    Geography_Germany: int   # True if living germany 
    Geography_Spain: int  # True if living in spain
    
    class Config:
        Schema_extra = {
            "example": {
                "CreditScore":610,  
                "Gender": 1,
                "Age": 27,
                "Tenure":2,
                "Balance": 1222.98, 
                "NumOfProducts": 2,
                "HasCrCard": 0,
                "IsActiveMember" :1,
                "EstimatedSalary": 122284.00,
                "Geography_Germany": 1, 
                "Geography_Spain": 0
                
            }        
           
        }
    
    
# Initalise FastAPI app

app = FastAPI(
    title = "Customer Churn Predictor",
    description = " Predicts if a Customer will Churn from key features" ,
    version =  "1.0.0" 
)

# Load the trained model

model_path = os.path.join("models", "churn_analysis_model.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)
    
with open(os.path.join("models", "model_scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)



@app.post("/predict")
def predict_churn(customer:Customer_Data):
    
    """
Predict the customers churn

    """
    # Convet the input to numpy array
    features = np.array([[
       customer.CreditScore,customer.Gender,
       customer.Age,customer.Tenure,
       customer.Balance,customer.NumOfProducts,
       customer.HasCrCard,customer.IsActiveMember,
       customer.EstimatedSalary,customer.Geography_Germany,
       customer.Geography_Spain,
    ]])
    
    # scale the data    
    scaled_features = scaler.transform(features)
    
    # MAKE Prediction
    prediction = model.predict(scaled_features)[0]
    
    # return 1 0r 0.
    # exited 1 stayed 0
    
    return {
        "churn_analysis":int(prediction)
    }



@app.get("/")
def health_check():
    return{"status": "healthy", "model": "Churn_analysis_model"}

