import numpy as np
import nibabel as nib
import tensorflow as tf
import shutil
from agnosticapi.server.models.cv_model.predict import cv_predict
from agnosticapi.server.models.seg3d_model.predict import seg3d_predict

class Model:
    def __init__(self, name, model_type, model_files, endpoint, task, description, ai_model_type, task_specific, ecology_specific,
                 language, dependencies, tool_url, last_update, license, contact_name, contact_email,
                 contact_responsiveness, parameters=None):
        self.name = name
        self.model_type = model_type
        self.model_files = model_files
        self.endpoint = endpoint
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
        self.model = tf.keras.models.load_model(model_path)

    def preprocess(self, file):
        raise NotImplementedError("This method should be overridden by subclasses")

    def predict(self, model_path, file):
        raise NotImplementedError("This method should be overridden by subclasses")

    def postprocess(self, output):
        raise NotImplementedError("This method should be overridden by subclasses")

    def retrain(self, model_path, file):
        raise NotImplementedError("This method should be overridden by subclasses")

    def save_output(self, output, output_path):
        np.save(output_path, output)

class CVModel(Model):

    def preprocess(self, file):
        img = np.uint8(tf.image.resize(tf.io.decode_image(file), (224, 224),
                                       method=tf.image.ResizeMethod.BILINEAR))
        img = img / 255 * 2 - 1  # normalize image
        return np.expand_dims(tf.image.img_to_array(img), axis=0)  # Add batch dimension

    def predict(self, model_path, file):
        return cv_predict(model_path, file)

    def postprocess(self, output):
        predicted_class = output[0].argmax()
        probability = output[0].max()
        return {"class": int(predicted_class), "probability": float(probability)}

class Seg3DModel(Model):
    def preprocess(self, uploaded_file):
        path = f"{uploaded_file.filename}"
        with open(path, 'w+b') as file:
            shutil.copyfileobj(uploaded_file.file, file)
        if path.endswith('.nii'):
            return path
        elif path.endswith('.npy'):
            npz_data = np.load(path)
            nifti_img = nib.Nifti1Image(npz_data['arr_0'], np.eye(4))
            nifti_path = path.replace('.npy', '.nii')
            nib.save(nifti_img, nifti_path)
            return nifti_path
        else:
            raise ValueError("Unsupported file format. Please provide a .nii or .npz file.")

    def predict(self, model_path, file_path):
        print('Trying to predict...')
        preprocessed_file_path = self.preprocess(file_path)
        labels = seg3d_predict(model_path, preprocessed_file_path)
        npy_path, nii_path = self.postprocess(labels, preprocessed_file_path.replace('.nii', '.npz'), preprocessed_file_path)
        return npy_path, nii_path

    def postprocess(self, output, output_path_npy, output_path_nii):
        # Save as .nii
        nib.save(nib.Nifti1Image(output, np.eye(4)), output_path_nii)
        # Save as .npy
        np.savez(output_path_npy, output)
        return output_path_npy, output_path_nii
