from fastapi import FastAPI, File, UploadFile
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import uvicorn
from PIL import Image
import io
import os
import shutil
import sys
import json
import uuid  # Import UUID library


## for seg3dtest
#import app.server.seg3d_backend.ls_seg3d as ls_seg3d             ### use for docker
import seg3d_backend.ls_seg3d as ls_seg3d
print("MADE IT HERE")
#import ls_seg3d

print("Tensorflow version: ", tf.__version__)
app = FastAPI() # Create a FastAPI instance that will be used to define the API

# Validating the user ID
def validate_uuid(uuid_str: str) -> bool:
    try:
        uuid.UUID(uuid_str)
        return True
    except ValueError:
        return False

def predict(img):
    # Load the MobileNetV2 model (assuming it's in the same directory)
    model = load_model('app/server/models/cv_model/MobileNetv2_model.keras')

    # Preprocess the image (resize, normalize etc.)
    img = np.uint8(tf.image.resize(tf.io.decode_image(img), (224, 224),
                                   method=tf.image.ResizeMethod.BILINEAR))
    img = img / 255 * 2 - 1        # normalize image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(x)

    # Return the top prediction (modify as needed)
    # Check classname below
    # https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
    return predictions[0].argmax(), predictions[0].max()

# FastAPI decorator to define the API endpoint
@app.post("/predict") # using an HTML method applied to a resource
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
async def seg3dtest(uploaded_file: UploadFile = File(...),
                    uuid: str = Header(None)):
    try:
        #if uuid is None or not validate_uuid(uuid):
        #    raise HTTPException(status_code=400, detail="Invalid or missing UUID")
        print("Received UUID server: ", str(uuid))
        # await uploaded_file.read()
        path = f"{uploaded_file.filename}"
        with open(path, 'w+b') as file:
            shutil.copyfileobj(uploaded_file.file, file)
        print(os.system('ls ./'))
        labels = ls_seg3d.ls_seg3d(path)
        labels = labels.astype(np.uint8)

        print("labels size info: ", labels.shape, labels.dtype, labels.nbytes)
        data = io.BytesIO(labels.tobytes())
        response = StreamingResponse(data, media_type = 'application/octet-stream')
        #request_uuid = uuid.uuid4()
        # Add the UUID as a custom header
        response.headers['X-Response-UUID'] = str(uuid)

        # Return the StreamingResponse object
        os.system("rm -r "+ path)
        return response
        '''
        # Generate a UUID for the request
        request_uuid = uuid.uuid4()

        # Return the streaming response with UUID
        return {
            "uuid": str(request_uuid),  # Convert UUID to string
            "response": response
        }
        '''
    except Exception as e:
        if os.path.exists('tmp'):
            os.system('rm -r tmp')
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)   # run on localhost port 8000 by default
