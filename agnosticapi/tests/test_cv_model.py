import pytest
import numpy as np
from agnosticapi.server.model_definitions import CVModel
from agnosticapi.server.models.cv_model import model_files

@pytest.fixture
def cv_model():
    return CVModel(
        name="CVModel",
        model_type="Package or Library",
        model_files=model_files,
        endpoint="/cv",
        task="Image Classification",
        description="MobileNetV2 image classification model",
        ai_model_type="Convolutional Neural Network",
        task_specific=True,
        ecology_specific=False,
        language="Python",
        dependencies=["tensorflow", "numpy"],
        tool_url="https://github.com/yourrepo/cvmodel",
        last_update="June 26, 2024",
        license="GNU General Public License",
        contact_name="Your Name",
        contact_email="your.email@example.com",
        contact_responsiveness="Very responsive"
    )

def test_cv_model_load(cv_model):
    model_path = cv_model.model_files
    cv_model.load(model_path)
    assert cv_model.model is not None

def test_cv_model_preprocess(cv_model):
    with open("abc_global_center_logo.jpeg", "rb") as f:
        img_bytes = f.read()
    preprocessed = cv_model.preprocess(img_bytes)
    assert preprocessed.shape == (1, 224, 224, 3) # TODO consider testing normalization function here

def test_cv_model_predict(cv_model):
    model_path = cv_model.model_files
    cv_model.load(model_path)
    with open("abc_global_center_logo.jpeg", "rb") as f:
        img_bytes = f.read()
    predictions = cv_model.predict(model_path, img_bytes)
    assert isinstance(predictions, np.ndarray)

def test_cv_model_postprocess(cv_model):
    output = np.random.rand(1, 1000)
    result = cv_model.postprocess(output)
    assert "class" in result
    assert "probability" in result
