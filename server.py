import os
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
import tensorflow as tf

# ✅ Suppress TensorFlow CPU feature warnings & oneDNN messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress unnecessary TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only usage

# ✅ Optimize CPU usage for TensorFlow
try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
except Exception as e:
    print(f"[WARNING] Error setting TensorFlow threading: {e}")

app = FastAPI()

# ✅ Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL if needed
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load the model once to prevent reloading on every request
def load_model():
    vgg_model_path = "models/vgg16_model.keras"
    rf_model_path = "models/rf_model.joblib"
    smote_data_path = "models/X_train_smote.npy"

    # Check if model files exist
    if not os.path.exists(vgg_model_path):
        raise FileNotFoundError(f"Model file missing: {vgg_model_path}")
    if not os.path.exists(rf_model_path):
        raise FileNotFoundError(f"Model file missing: {rf_model_path}")
    if not os.path.exists(smote_data_path):
        raise FileNotFoundError(f"SMOTE dataset missing: {smote_data_path}")

    return Model(vgg_model_path, rf_model_path, smote_data_path)

model = load_model()

# ✅ Disable interactive mode in matplotlib to prevent figure window issues
plt.ioff()

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
        if model is None:
            return JSONResponse(status_code=500, content={"error": "Model not loaded properly."})

        # Load the image
        img_bytes = await image.read()
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        
        # Convert image to format suitable for prediction (NumPy array)
        image_cv = np.array(img)
        
        # Make the prediction
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
        
        return {
            "prediction": prediction,
            "plots": plot_base64_list  # Return the plots as base64 strings
        }
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
