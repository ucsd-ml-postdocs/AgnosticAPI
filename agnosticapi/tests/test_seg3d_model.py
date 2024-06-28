import pytest
import os
import numpy as np
from fastapi import UploadFile
from agnosticapi.server.model_definitions import Seg3DModel
from agnosticapi.server.models.seg3d_model import model_files

@pytest.fixture
def seg3d_model():
    return Seg3DModel(
        name="Seg3DModelV1",
        model_type="Package or Library",
        model_files=model_files,
        endpoint="/seg3d",
        task="Image Segmentation",
        description="3D segmentation model to segment medical imagery of the heart.",
        ai_model_type="Convolutional Neural Network",
        task_specific=True,
        ecology_specific=False,
        language="Python",
        dependencies=["tensorflow", "numpy"],
        tool_url="https://github.com/yourrepo/seg3DmodelV1",
        last_update="June 26, 2024",
        license="GNU General Public License",
        contact_name="Lauren Severance",
        contact_email="your.email@example.com",
        contact_responsiveness="Very responsive"
    )

@pytest.fixture
def uploaded_file():
    file_path = "15_NRR-ID27468.nii"
    return UploadFile(filename=file_path, file=open(file_path, "rb"))

def test_seg3d_model_load(seg3d_model):
    model_path = seg3d_model.model_files
    seg3d_model.load(model_path)
    assert seg3d_model.model is not None

def test_seg3d_model_preprocess(seg3d_model, uploaded_file):
    preprocessed_path = seg3d_model.preprocess(uploaded_file)
    assert preprocessed_path.endswith('.nii')
    assert os.path.exists(preprocessed_path)

def test_seg3d_model_predict(seg3d_model, uploaded_file):
    model_path = seg3d_model.model_files
    seg3d_model.load(model_path)
    npy_path, nii_path = seg3d_model.predict(model_path, uploaded_file)
    assert os.path.exists(npy_path)
    assert os.path.exists(nii_path)

def test_seg3d_model_postprocess(seg3d_model):
    output = np.random.rand(200, 200, 160)
    output_path_npy = "agnosticapi/test-outputs/output_path.npz"
    output_path_nii = "agnosticapi/test-outputs/output_path.nii"
    npy_path, nii_path = seg3d_model.postprocess(output, output_path_npy, output_path_nii)
    assert os.path.exists(npy_path)
    assert os.path.exists(nii_path)
