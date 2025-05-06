import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from model_utils import load_model



# Initialize FastAPI app
app = FastAPI()


# Define the request body
class ChurnRequest(BaseModel):
    features: list  # expecting [Credit score , Age , Tenure , Balance , NumOfProducts , HasCrCard , IsActiveMember , EstimatedSalary ,  gender , Geography_France , Geography_Germany , Geography_Spain]
    
# Define a route
@app.post("/predict")
def predict_Churn(request: ChurnRequest):
    # Convert list to 2D numpy array
    features = np.array(request.features).reshape(1, -1)
    model=load_model()
    pred_class_idx = model.predict(features)[0]
    if int(pred_class_idx) == 1 :
        pred_res="Exited"
    else:
        pred_res="Remained"
    return {"prediction": pred_res}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
#     "features": [619 , 42 , 2 , 0.00 , 1 , 1 , 1 , 101348.88 , 0 , 1.0 , 0.0 , 0.0]