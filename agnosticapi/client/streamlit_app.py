import os
import streamlit as st
import numpy as np
from segmentation_client import upload_for_segmentation

from agnosticapi.server.media import dir as media_dir
from agnosticapi.server.models import CVModel, Seg3DModel

# Dynamically determine the path to the logo
logo_path = os.path.join(media_dir, 'DALLE_JKB_logo.png')

# Verify the path
if not os.path.isfile(logo_path):
    st.error(f"Logo file not found at {logo_path}")
else:
    st.image(logo_path, width=300)  # Adjust the width as needed

# Function to handle file selection and initiate segmentation
def start_segmentation(file, server_url):
    if file is not None:
        mask = upload_for_segmentation(file, server_url=server_url)
        if mask is not None:
            st.success("Model run completed!")
            np.save('agnosticapi/client/Predictions/mask.npy', mask)
            st.write("Prediction output generated and saved.")
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
file = st.file_uploader("Select Data File", type=["nii"])

# Start segmentation button
if st.button("Start Segmentation"):
    start_segmentation(file, server_url)
