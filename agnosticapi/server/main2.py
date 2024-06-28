from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from agnosticapi.server.model_definitions import CVModel, Seg3DModel
from agnosticapi.server.models.cv_model import model_files as cv_model_files, model_history as cv_model_history, code_dir as cv_code_dir
from agnosticapi.server.models.seg3d_model import model_files as seg3d_model_files, model_history as seg3d_model_history, code_dir as seg3d_code_dir

import numpy as np
import uvicorn

app = FastAPI()

# Initialize and load models
cv_model = CVModel(
    name="CVModel",
    model_type="Package or Library",
    model_files=cv_model_files,
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

seg3d_model = Seg3DModel(
    name="Seg3DModelV1",
    model_type="Package or Library",
    model_files=seg3d_model_files,
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

@app.post("/cv")
async def prediction(model_path: str = Form(...), file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        predictions = cv_model.predict(model_path, image_bytes)
        result = cv_model.postprocess(predictions)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/seg3d")
async def seg3dtest(model_path: str = Form(...), uploaded_file: UploadFile = File(...), uuid: str = Header(None)):
    try:
        response = seg3d_model.predict(model_path, uploaded_file, uuid)
        return response
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
