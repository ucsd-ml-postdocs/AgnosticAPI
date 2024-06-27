import requests
import uuid
import numpy as np  # Optional, for processing response data

def upload_for_segmentation(file, server_url="http://localhost:8000/seg3dtest/"):
    """
    Uploads a medical data file for segmentation and retrieves the segmentation mask.

    Args:
        file (File-like object): Medical data file (.nii format).
        server_url (str, optional): URL of the segmentation server. Defaults to "http://localhost:8000/seg3dtest/".

    Returns:
        numpy.ndarray: Segmentation mask as a NumPy array (if successful), None otherwise.
    """

    # Generate a UUID for the request
    request_uuid = uuid.uuid4()

    # Open the file in binary mode
    try:
        files = {'uploaded_file': file}

        # Include the UUID in the request headers
        headers = {'uuid': str(request_uuid)}

        print(f"Sent UUID client: {str(request_uuid)}")

        # Send the request with the file and headers
        response = requests.post(server_url, files=files, headers=headers)

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None

    # Check if the upload was successful
    if response.status_code != 200:
        print(f"Failed to upload file: {response.status_code}")
        print(f"Response body: {response.text}")  # Print detailed error message
        return None

    # Extract the UUID from the response headers
    response_uuid = response.headers.get('X-Response-UUID')

    # Check if the UUID was received
    if response_uuid is not None:
        print(f"Received UUID client: {str(response_uuid)}")

        # Process the response data
        data = response.content
        try:
            # Assuming response is a raw byte stream representing the mask
            mask = np.frombuffer(data, dtype=np.uint8).reshape((200, 200, 160))  # Adjust data type according to server response format
            return mask
        except Exception as e:
            print(f"Error processing response data: {e}")
            return None

    else:
        print("Response UUID not found in response headers.")
        return None
