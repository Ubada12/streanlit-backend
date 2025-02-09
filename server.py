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

# ✅ Limit TensorFlow GPU memory growth (IMPORTANT FOR LOW-RAM SERVERS)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# ✅ Disable TensorFlow optimizations that consume extra memory
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ✅ Force TensorFlow to use only the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

# ✅ Optimize CPU usage for TensorFlow
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# ✅ Set a memory limit (e.g., 512MB) if needed
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)]
        )
    except RuntimeError as e:
        print(e)

app = FastAPI()

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL if needed
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
def load_model():
    return Model('models/vgg16_model.keras', 'models/rf_model.joblib', 'models/X_train_smote.npy')

# Disable interactive mode in matplotlib to prevent figure window issues
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
        return {
            "prediction": prediction,
            "plots": plot_base64_list  # Return the plots as base64 strings
        }
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
