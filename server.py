import uvicorn
from fastapi import FastAPI, File, UploadFile
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

# Load the model
def load_model():
    model_path = os.path.abspath('models/vgg16_model.keras')
    print("Loading model from:", model_path)  # Debugging
    print("Current working directory:", os.getcwd())
    return Model('models/vgg16_model.keras', 'models/rf_model.joblib', 'models/X_train_smote.npy')

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
        # Save the figure to a BytesIO buffer
        buf = io.BytesIO()
        plot.savefig(buf, format='png')
        buf.seek(0)

        # Convert the buffer to a base64 string
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plot_base64_list.append(plot_base64)
    
    # Prepare the response
    result = {
        "prediction": prediction,
        "plots": plot_base64_list  # Return the plots as base64 strings
    }
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
