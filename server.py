import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import numpy as np
from models.pipeline import Model
from PIL import Image
import base64
import io
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse
import os

app = FastAPI()

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify React app URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths
VGG_MODEL_PATH = "models/vgg16_model.keras"
RF_MODEL_PATH = "models/rf_model.joblib"
SMOTE_DATA_PATH = "models/X_train_smote.npy"

def load_model():
    """Load model if the file exists, else raise an error."""
    if not os.path.exists(VGG_MODEL_PATH):
        raise FileNotFoundError(f"🚨 Model file missing: {VGG_MODEL_PATH}. Ensure it exists!")
    if not os.path.exists(RF_MODEL_PATH):
        raise FileNotFoundError(f"🚨 Model file missing: {RF_MODEL_PATH}. Ensure it exists!")
    if not os.path.exists(SMOTE_DATA_PATH):
        raise FileNotFoundError(f"🚨 Data file missing: {SMOTE_DATA_PATH}. Ensure it exists!")
    
    return Model(VGG_MODEL_PATH, RF_MODEL_PATH, SMOTE_DATA_PATH)

# Disable interactive mode in matplotlib to prevent opening of figure windows
plt.ioff()

# Define API routes
@app.get("/")
def read_home():
    return {"message": "Welcome to the FastAPI Server!"}

@app.post("/predictions")
async def predict(
    image: UploadFile = File(...),
    latitude: float = 20.0,
    longitude: float = 20.0
):
    try:
        # Load the image
        img_bytes = await image.read()
        img = Image.open(BytesIO(img_bytes))
        
        # Convert image to format suitable for prediction (NumPy array)
        image_cv = np.array(img)
        
        # Load the model and make the prediction
        model = load_model()
        prediction, plot_list = model.predict(image_cv, latitude, longitude)
        
        # Convert prediction to native Python int to ensure JSON serializability
        prediction = int(prediction)

        # Convert plots to base64-encoded strings for JSON serialization
        plot_base64_list = []
        for plot in plot_list:
            buf = io.BytesIO()
            plot.savefig(buf, format='png')
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plot_base64_list.append(plot_base64)
        
        # Prepare the response
        return {"prediction": prediction, "plots": plot_base64_list}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
