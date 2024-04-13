import os
import shutil

from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image
import io

## for seg3dtest
import app.ls_seg3d as ls_seg3d

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
        ls_seg3d.ls_seg3d(path)
        # ls_seg3d.ls_seg3d('/15_NRR-ID27468.nii.gz')
        # ls_seg3d.ls_seg3d('/Users/zhengzeng/PycharmProjects/AgnosticAPI/15_NRR-ID27468.nii.gz')
        os.system(f'rm {path}')
        return {"status": "success!"}
    except Exception as e:
        if os.path.exists('tmp'):
            os.system('rm -r tmp')
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
