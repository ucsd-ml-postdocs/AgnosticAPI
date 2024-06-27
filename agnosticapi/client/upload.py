import requests
import uuid
import numpy as np

def upload_for_prediction(file, server_url="http://localhost:8000/seg3d"):
    """
    Uploads a data file for prediction and retrieves the response.

    Args:
        file (File-like object): Data file.
        server_url (str, optional): URL of the prediction server. Defaults to "http://localhost:8000/seg3d".

    Returns:
        dict or numpy.ndarray: Prediction result as a dictionary or a segmentation mask as a NumPy array.
    """
    # Generate a UUID for the request
    request_uuid = uuid.uuid4()

    try:
        files = {'uploaded_file': file}
        headers = {'uuid': str(request_uuid)}
        print(f"Sent UUID client: {str(request_uuid)}")

        # Send the request with the file and headers
        response = requests.post(server_url, files=files, headers=headers)
    except FileNotFoundError:
        print(f"Error: File not found: {file}")
        return None

    if response.status_code != 200:
        print(f"Failed to upload file: {response.status_code}")
        print(f"Response body: {response.text}")
        return None

    response_uuid = response.headers.get('X-Response-UUID')
    if response_uuid:
        print(f"Received UUID client: {str(response_uuid)}")

    if server_url.endswith("/seg3d") or server_url.endswith("/seg3d/"):
        # Process segmentation response
        data = response.content
        try:
            mask = np.frombuffer(data, dtype=np.uint8).reshape((200, 200, 160))
            return mask
        except Exception as e:
            print(f"Error processing response data: {e}")
            return None
    else:
        # Process generic prediction response
        try:
            return response.json()
        except Exception as e:
            print(f"Error processing response data: {e}")
            return None
