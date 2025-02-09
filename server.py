import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import numpy as np
from models.pipeline import Model
from PIL import Image
import base64
import io
import os
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse

app = FastAPI()

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify React app URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure models directory exists
MODEL_DIR = os.path.abspath("models")
VGG_MODEL_PATH = os.path.join(MODEL_DIR, "vgg16_model.keras")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
SMOTE_PATH = os.path.join(MODEL_DIR, "X_train_smote.npy")

# Load the model
def load_model():
    # Check if the model files exist before loading
    if not os.path.exists(VGG_MODEL_PATH):
        raise HTTPException(status_code=500, detail=f"Model file not found: {VGG_MODEL_PATH}")
    if not os.path.exists(RF_MODEL_PATH):
        raise HTTPException(status_code=500, detail=f"RF Model file not found: {RF_MODEL_PATH}")
    if not os.path.exists(SMOTE_PATH):
        raise HTTPException(status_code=500, detail=f"SMOTE data file not found: {SMOTE_PATH}")

    print("✅ Model files found, loading...")
    try:
        model = Model(VGG_MODEL_PATH, RF_MODEL_PATH, SMOTE_PATH)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Disable interactive mode in matplotlib to prevent opening figure windows
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
        # Read image from request
        img_bytes = await image.read()
        img = Image.open(BytesIO(img_bytes))
        
        # Convert image to numpy array
        image_cv = np.array(img)

        # Load model and make prediction
        model = load_model()
        prediction, plot_list = model.predict(image_cv, latitude, longitude)

        # Ensure prediction is JSON serializable
        prediction = int(prediction)

        # Convert plots to base64-encoded strings
        plot_base64_list = []
        for plot in plot_list:
            buf = io.BytesIO()
            plot.savefig(buf, format='png')
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plot_base64_list.append(plot_base64)

        # Return response
        return JSONResponse(content={
            "prediction": prediction,
            "plots": plot_base64_list
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
