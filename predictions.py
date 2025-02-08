import streamlit as st
import requests
from PIL import Image
import io

# Function to send image and coordinates to FastAPI backend
def get_prediction(image, latitude, longitude):
    url = "http://localhost:8000/predictions"
    
    # Prepare the image for sending
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Create the payload (image data and coordinates)
    files = {'image': ('image.png', img_byte_arr, 'image/png')}
    data = {'latitude': latitude, 'longitude': longitude}

    # Send POST request to FastAPI
    response = requests.post(url, files=files, data=data)
    
    # Return prediction result
    return response.json()

# Front-end for the Streamlit app
def app():
    st.title("Flood Prediction Using Image and Weather Data")

    # Upload Image
    st.subheader("Upload Image:")
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Convert the image to an array
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Input for Latitude and Longitude
    st.subheader("Enter Latitude and Longitude:")
    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=20.0)
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=20.0)

    if uploaded_image is not None:
        # Send the image and coordinates to FastAPI for prediction
        prediction_result = get_prediction(image, latitude, longitude)

        # Display the result
        st.write("Prediction Result:", "Flood Detected" if prediction_result["prediction"] == 1 else "No Flood Detected")

        # Display plots (if available)
        for plot in prediction_result["plots"]:
            st.pyplot(plot)

if __name__ == "__main__":
    app()
