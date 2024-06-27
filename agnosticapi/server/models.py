from tensorflow.keras.models import load_model
import numpy as np
import os
import uuid

class Model:
    def __init__(self, name, model_type, task, description, ai_model_type, task_specific, ecology_specific,
                 language, dependencies, tool_url, last_update, license, contact_name, contact_email,
                 contact_responsiveness, parameters=None):
        self.name = name
        self.model_type = model_type
        self.task = task
        self.description = description
        self.ai_model_type = ai_model_type
        self.task_specific = task_specific
        self.ecology_specific = ecology_specific
        self.language = language
        self.dependencies = dependencies
        self.tool_url = tool_url
        self.last_update = last_update
        self.license = license
        self.contact_name = contact_name
        self.contact_email = contact_email
        self.contact_responsiveness = contact_responsiveness
        self.parameters = parameters if parameters else {}

    def load(self, model_path):
        self.model = load_model(model_path)

    def preprocess(self, file):
        raise NotImplementedError("This method should be overridden by subclasses")

    def predict(self, data):
        raise NotImplementedError("This method should be overridden by subclasses")

    def save_output(self, output, output_path):
        np.save(output_path, output)

class CVModel(Model):
    

    def preprocess(self, file):
        img = np.uint8(tf.image.resize(tf.io.decode_image(file), (224, 224),
                                       method=tf.image.ResizeMethod.BILINEAR))
        img = img / 255 * 2 - 1  # normalize image
        return np.expand_dims(image.img_to_array(img), axis=0)  # Add batch dimension

    def predict(self, data):
        return self.model.predict(data)

class Seg3DModel(Model):

    from agnosticapi.server.ls_seg3d_model.seg3d_backend import ls_seg3d
    def preprocess(self, file_path):
        
        pass

    def predict(self, data):
        # Custom prediction for segmentation
        labels = ls_seg3d.ls_seg3d()
        pass
