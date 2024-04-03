from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image
import io
print(tf.__version__)
app = FastAPI()

# Load the MobileNetV2 model (assuming it's in the same directory)
model = load_model('app/model/MobileNetv2_model.keras')

# Define the prediction function (replace with your actual logic)
def predict(img):
  # Preprocess the image (resize, normalize etc.)
  # img = image.load_img(img, target_size=(224, 224))  # Adjust based on model input size
  img = np.uint8(tf.image.resize(tf.io.decode_image(img), (224, 224),
                                 method=tf.image.ResizeMethod.BILINEAR))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)  # Add batch dimension
  # x = preprocess_input(x)  # Assuming you have a preprocess_input function

  # Make prediction
  predictions = model.predict(x)

  # Return the top prediction (modify as needed)
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

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)

