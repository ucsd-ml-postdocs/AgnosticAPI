import os
import streamlit as st
import numpy as np
from upload import upload_for_prediction
from agnosticapi.server.media import dir
from agnosticapi.server.main2 import cv_model, seg3d_model

# Dynamically determine the path to the logo
logo_path = os.path.join(dir, 'DALLE_JKB_logo.png')

# Verify the path
if not os.path.isfile(logo_path):
    st.error(f"Logo file not found at {logo_path}")
else:
    st.image(logo_path, width=300)  # Adjust the width as needed

# Function to handle file selection and initiate prediction
def start_prediction(file, model_path, server_url):
    if file is not None:
        content_type, result = upload_for_prediction(file, model_path, server_url=server_url)
        if result is not None:
            if content_type and 'application/octet-stream' in content_type:
                st.success("Segmentation model run completed!")
                np.save('agnosticapi/client/Predictions/mask.npy', result)
                st.write("Segmentation output generated and saved.")
            elif content_type and 'application/json' in content_type and isinstance(result, dict):
                if 'error' not in result:
                    st.success("Prediction model run completed!")
                    st.json(result)
                else:
                    st.error(f"Prediction failed with error: {result['error']}")
            elif isinstance(result, str):  # Handle string error messages
                st.error(result)
            else:
                st.error("Unsupported response format.")
        else:
            st.error("Error during upload or processing.")
    else:
        st.error("No file selected.")

# Title and description
st.title("Agnostic API")
st.write("Select a data file and kick off your model.")

base_server_url = "http://localhost:8000"

# Model and Server URL selection
model_options = {
    "Image Classification (CVModel)": (cv_model.endpoint, cv_model.model_files),
    "3D Segmentation (Seg3DModelV1)": (seg3d_model.endpoint, seg3d_model.model_files),
    "Custom": (None, None)
}
model_choice = st.selectbox("Select Model", list(model_options.keys()))

endpoint, model_path = model_options[model_choice]
server_url = f"{base_server_url}{endpoint}" if endpoint else None

# Custom URL entry
if model_choice == "Custom":
    server_url = st.text_input("Enter Custom URL")
    model_path = st.text_input("Enter Custom Model Path")

# File selection
file = st.file_uploader("Select Data File", type=["nii", "jpg", "png", "jpeg"])

# Start prediction button
if st.button("Start Prediction"):
    start_prediction(file, model_path, server_url)
