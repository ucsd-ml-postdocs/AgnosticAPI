import requests
import uuid
import numpy as np

def upload_for_prediction(file, model_path, server_url="http://localhost:8000/seg3d"):
    """
    Uploads a data file for prediction and retrieves the response.

    Args:
        file (File-like object): Data file.
        model_path (str): Path to the model.
        server_url (str, optional): URL of the prediction server. Defaults to "http://localhost:8000/seg3d".

    Returns:
        tuple: Response content type and data as a dictionary or a segmentation mask as a NumPy array, or an error message.
    """
    # Generate a UUID for the request
    request_uuid = uuid.uuid4()

    try:
        files = {'uploaded_file': file}
        data = {'model_path': model_path}
        headers = {'uuid': str(request_uuid)}
        print(f"Sent UUID client: {str(request_uuid)}")

        # Send the request with the file, model path, and headers
        response = requests.post(server_url, files=files, data=data, headers=headers)
    except FileNotFoundError:
        print(f"Error: File not found: {file}")
        return None, "File not found"

    if response.status_code != 200:
        print(f"Failed to upload file: {response.status_code}")
        print(f"Response body: {response.text}")
        return None, f"Failed to upload file: {response.status_code} - {response.text}"

    response_uuid = response.headers.get('X-Response-UUID')
    if response_uuid:
        print(f"Received UUID client: {str(response_uuid)}")

    content_type = response.headers.get('Content-Type')

    if 'application/octet-stream' in content_type:
        # Process segmentation response
        data = response.content
        try:
            mask = np.frombuffer(data, dtype=np.uint8).reshape((200, 200, 160))
            return content_type, mask
        except Exception as e:
            print(f"Error processing response data: {e}")
            return content_type, f"Error processing response data: {e}"
    else:
        # Process generic prediction response
        try:
            return content_type, response.json()
        except Exception as e:
            print(f"Error processing response data: {e}")
            return content_type, f"Error processing response data: {e}"
