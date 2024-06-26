import streamlit as st
import numpy as np
from segmentation_client import upload_for_segmentation

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
st.title("Segmentation Client")
st.write("Select a medical data file and initiate segmentation.")

# Server URL selection
server_url_options = [
    "http://localhost:80/seg3dtest/",
    'http://132.239.113.208:80/seg3dtest/',
    "http://your-server2.com:5000/segmentation",
    "custom"
]
server_url = st.selectbox("Server URL", server_url_options)

# Custom URL entry
if server_url == "custom":
    server_url = st.text_input("Enter Custom URL")

# File selection
file = st.file_uploader("Select Medical Data File", type=["nii"])

# Start segmentation button
if st.button("Start Segmentation"):
    start_segmentation(file, server_url)
