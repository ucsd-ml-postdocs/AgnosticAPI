import requests
import numpy as np
import glob
url = 'http://localhost:8000/seg3dtest/'
file_path = '/home/harp/Desktop/15_NRR-ID27468.nii'
file_path = '/Users/joe/Downloads/15_NRR-ID27468.nii'

print(glob.glob(file_path))

#open the file in binary mode
with open(file_path, 'rb') as f:
    files = {'uploaded_file': f}
    print("files", files)
    response = requests.post(url, files=files)


# Check if the upload was successful
if response.status_code != 200:
    print("failed to upload file: ", response.status_code)
    print("Response body: ", response.text) # print the detailed error message
else:
    data = response.content
    mask = np.frombuffer(data, dtype=np.uint8).reshape((200,200,160))
    print(mask)
