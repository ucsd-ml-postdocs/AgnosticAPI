from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import StreamingResponse
from agnosticapi.server.models import CVModel, Seg3DModel  # Import the model classes
import numpy as np
import shutil
import io
import os

app = FastAPI()

# Define models
cv_model = CVModel(
    name="CVModel",
    model_type="Package or Library",
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
cv_model.load('agnosticapi/server/models/cv_model/MobileNetv2_model.keras')

Seg3D_model_V1 = Seg3DModel(
    name="Seg3DModelV1",
    model_type="Package or Library",
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
from agnosticapi.server.models.ls_seg3d_model import model_files
Seg3D_model_V1.load(model_files)


@app.post("/cv")
async def prediction(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        data = cv_model.preprocess(image_bytes)
        predictions = cv_model.predict(data)
        predicted_class, probability = predictions[0].argmax(), predictions[0].max()
        return {"class": int(predicted_class), "probability": float(probability)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/seg3d")
async def seg3dtest(uploaded_file: UploadFile = File(...), uuid: str = Header(None)):
    try:
        path = f"{uploaded_file.filename}"
        with open(path, 'w+b') as file:
            shutil.copyfileobj(uploaded_file.file, file)
        labels = Seg3D_model_V1.predict(path)
        labels = labels.astype(np.uint8)
        data = io.BytesIO(labels.tobytes())
        response = StreamingResponse(data, media_type='application/octet-stream')
        response.headers['X-Response-UUID'] = str(uuid)
        os.system("rm -r " + path)
        return response
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
