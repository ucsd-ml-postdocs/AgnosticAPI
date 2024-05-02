import os
import shutil
import sys

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
## for seg3dtest
#import app.ls_seg3d as ls_seg3d
import ls_seg3d as ls_seg3d

print(tf.__version__)
app = FastAPI()


# Define the prediction function (replace with your actual logic)
def predict(img):
    # Load the MobileNetV2 model (assuming it's in the same directory)
    model = load_model('app/model/cv_model/MobileNetv2_model.keras')

    # Preprocess the image (resize, normalize etc.)
    img = np.uint8(tf.image.resize(tf.io.decode_image(img), (224, 224),
                                   method=tf.image.ResizeMethod.BILINEAR))
    img = img / 255 * 2 - 1
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(x)

    # Return the top prediction (modify as needed)
    # Check classname below
    # https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
    return predictions[0].argmax(), predictions[0].max()

def convert_to_JSON(labels, filename):
    # Convert the NumPy array to a Python list of lists
    list_of_frames = labels.tolist()

    # Write the list of frames to a JSON file
    with open(filename, 'w') as json_file:
        json.dump(list_of_frames, json_file)
    return filename

@app.post("/predict")
async def prediction(file: UploadFile = File(...)):
    try:
        # Read the image data
        image_bytes = await file.read()

        # Make prediction
        predicted_class, probability = predict(image_bytes)

        # Return the prediction results
        return {
            "class": int(predicted_class),
            "probability": float(probability)
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/seg3dtest")
async def seg3dtest(uploaded_file: UploadFile = File(...)):
    try:
        # await uploaded_file.read()
        path = f"{uploaded_file.filename}"
        with open(path, 'w+b') as file:
            shutil.copyfileobj(uploaded_file.file, file)
        print(os.system('ls ./'))
        labels = ls_seg3d.ls_seg3d(path)
        labels = labels.astype(np.uint8)

        print("labels size info: ", labels.shape, labels.dtype, labels.nbytes)
        """

        json_filename = convert_to_JSON(labels,'output.json')

        # ls_seg3d.ls_seg3d('/15_NRR-ID27468.nii.gz')
        # ls_seg3d.ls_seg3d('/Users/zhengzeng/PycharmProjects/AgnosticAPI/15_NRR-ID27468.nii.gz')
        os.system(f'rm {path}')
        #return {"status": "success!"}
        #return json_file
        with open(json_filename, 'r') as json_file:
            json_contents = json.load(json_file)
        print("json_contents: ", type(json_contents))
        os.remove(json_filename)  # Optionally remove the file after reading its contents
        json_contents_size = sys.getsizeof(json_contents)
        print("Size of json_contents (in bytes):", json_contents_size)
        """
        frames_dict = {}

        # Loop over the third dimension of the array
        for i in range(labels.shape[2]):
            # Extract the frame and assign it as a value to the corresponding key
            frames_dict[f'frame_{i}'] = labels[:, :, i].tolist()
        print("frames_dict: ", type(frames_dict), sys.getsizeof(frames_dict))

        #return JSONResponse(content = jsonable_encoder(frames_dict))
        data = io.BytesIO(labels.tobytes())
        response = StreamingResponse(data, media_type = 'application/octet-stream')
        return response
    
    except Exception as e:
        if os.path.exists('tmp'):
            os.system('rm -r tmp')
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
