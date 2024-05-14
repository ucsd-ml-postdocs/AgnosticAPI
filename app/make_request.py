import requests
import numpy as np
import glob
import uuid

url = 'http://localhost:8000/seg3dtest/'
url = 'http://132.239.113.208:80/seg3dtest/'
#file_path = '/home/harp/Desktop/15_NRR-ID27468.nii'
file_path = '/Users/joe/Downloads/15_NRR-ID27468.nii'

print(glob.glob(file_path))

# Generate a UUID for the request
request_uuid = uuid.uuid4()

# Open the file in binary mode
with open(file_path, 'rb') as f:
    files = {'uploaded_file': f}

    # Include the UUID in the request headers
    headers = {'uuid': str(request_uuid)}

    print("Sent UUID client: ", str(request_uuid))
    # Send the request with the file and headers
    response = requests.post(url, files=files, headers=headers)

# Check if the upload was successful
if response.status_code != 200:
    print("Failed to upload file: ", response.status_code)
    print("Response body: ", response.text)  # Print the detailed error message
else:
    # Extract the UUID from the response headers
    response_uuid = response.headers.get('X-Response-UUID')

    # Check if the UUID was received
    if response_uuid is not None:
        print("Received UUID client :", str(response_uuid))

        # Process the response data
        data = response.content
        mask = np.frombuffer(data, dtype=np.uint8).reshape((200, 200, 160))
        print(mask)
        np.save('app/Predictions/mask.npy', mask)
    else:
        print("Response UUID not found in response headers.")
