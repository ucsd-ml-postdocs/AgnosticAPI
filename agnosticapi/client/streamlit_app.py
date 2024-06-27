import os
import streamlit as st
import numpy as np
from upload import upload_for_prediction
from agnosticapi.server.media import dir

# Dynamically determine the path to the logo
logo_path = os.path.join(dir, 'DALLE_JKB_logo.png')

# Verify the path
if not os.path.isfile(logo_path):
    st.error(f"Logo file not found at {logo_path}")
else:
    st.image(logo_path, width=300)  # Adjust the width as needed

# Function to handle file selection and initiate prediction
def start_prediction(file, server_url):
    if file is not None:
        result = upload_for_prediction(file, server_url=server_url)
        if result is not None:
            if isinstance(result, np.ndarray):
                st.success("Segmentation model run completed!")
                np.save('agnosticapi/client/Predictions/mask.npy', result)
                st.write("Segmentation output generated and saved.")
            else:
                st.success("Prediction model run completed!")
                st.json(result)
        else:
            st.error("Error during upload or processing.")
    else:
        st.error("No file selected.")

# Title and description
st.title("Agnostic API")
st.write("Select a data file and kick off your model.")

# Server URL selection
server_url_options = [
    "http://localhost:8000/seg3d/",
    "http://localhost:8000/cv/",
    "custom"
]
server_url = st.selectbox("Server URL", server_url_options)

# Custom URL entry
if server_url == "custom":
    server_url = st.text_input("Enter Custom URL")

# File selection
file = st.file_uploader("Select Data File", type=["nii", "jpg", "png", "jpeg"])

# Start prediction button
if st.button("Start Prediction"):
    start_prediction(file, server_url)
