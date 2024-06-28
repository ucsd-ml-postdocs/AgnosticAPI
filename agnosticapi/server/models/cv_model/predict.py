import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def cv_predict(model_path, img):
    # Load the MobileNetV2 model from the provided model path
    model = load_model(model_path)

    # Preprocess the image (resize, normalize etc.)
    img = np.uint8(tf.image.resize(tf.io.decode_image(img), (224, 224),
                                   method=tf.image.ResizeMethod.BILINEAR))
    img = img / 255 * 2 - 1  # normalize image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(x)

    # Return the top prediction
    return predictions
