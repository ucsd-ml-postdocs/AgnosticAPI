import os
import streamlit as st
import numpy as np
from segmentation_client import upload_for_segmentation

# Dynamically determine the path to the logo
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, '..', '..', 'media', 'DALLE_JKB_logo.png')

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
            st.success("Segmentation completed!")
            np.save('app/client/Predictions/mask.npy', mask)
            st.write("Segmentation mask saved.")
        else:
            st.error("Error during upload or processing.")
    else:
        st.error("No file selected.")

# Title and description
st.title("Agnostic API")
st.write("Select a data file and kick off your model.")

# Server URL selection
server_url_options = [
    "http://localhost:8000/segment3D/",
    "http://132.239.113.208:8000/segment3D/",
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
