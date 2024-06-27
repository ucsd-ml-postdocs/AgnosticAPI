# AgnosticAPI

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

A tool for performing image classification and 3D segmentation using pre-trained AI models.

## Directory Structure

```
AGNOSTICAPI
│
├── __pycache__                # Cached bytecode files for faster execution.
│
├── client                     # Client-side code for the application.
│   ├── __pycache__            # Cached bytecode files for faster execution.
│   ├── Predictions            # Directory to store prediction results.
│   │   └── mask.npy           # Numpy file containing the prediction mask.
│   ├── __init__.py            # Initialization file for the client module.
│   ├── graphical.py           # Script for the GUI client using tkinter for file selection and segmentation.
│   ├── streamlit_app.py       # Streamlit script to replace graphical.py for a web-based interface.
│   └── segmentation_client.py # Script to handle file uploads for segmentation and prediction.
│
├── server                     # Server-side code for the application.
│   ├── __pycache__            # Cached bytecode files for faster execution.
│   ├── media                  # Directory for storing media files.
│   ├── models                 # Directory to store the ML models.
│   │   ├── __pycache__        # Cached bytecode files for faster execution.
│   │   ├── cv_model           # Computer vision model directory.
│   │   │   └── MobileNetv2_model.keras # Pre-trained MobileNetV2 model file.
│   │   ├── ls_seg3d_model     # Directory for 3D segmentation model.
│   │   │   └── m20230623-163203wh500epochs # Pre-trained 3D segmentation model file.
│   │   ├── __init__.py        # Initialization file for the models module.
│   ├── __init__.py            # Initialization file for the server module.
│   ├── main.py                # Main script for starting the FastAPI server.
│   ├── main2.py               # Secondary main script, possibly for a different configuration or purpose.
│   ├── models.py              # Definition of model classes with metadata and processing methods.
│
├── agnosticapi.egg-info       # Metadata directory for the agnosticapi package.
│
├── venv                       # Virtual environment directory.
│
├── .gitignore                 # Git ignore file to exclude files from version control.
├── 15_NRR-ID27468.nii         # Sample NIfTI file for testing.
├── Dockerfile                 # Docker configuration file for containerizing the application.
├── EcoVizAPI_seg3d_joe_contract.docx  # EcoViz API contract with rules for engagement.
├── pyproject.toml             # Configuration file for building and packaging the project.
├── README.md                  # Readme file containing the project overview and instructions.
├── requirements.txt           # List of Python dependencies for the project.
├── seg3DenvMini.yml           # Minimal Conda environment configuration file.
├── setup.py                   # Script for installing the agnosticapi package.
└── view_mask.ipynb            # Jupyter notebook for visualizing prediction masks.
```

## Installation

### Using Docker (Recommended for Windows)
```bash
docker build -t mobile-net-app .
docker run -p 80:80 mobile-net-app
```

### Using Conda (Recommended for Mac with Mac chip)
From within the AgnosticAPI directory, please run (line by line):
```bash
conda env create -n apienv python=3.8
conda activate apienv
conda env update -f seg3DenvMini.yml 
```
Note: You may have to use `conda` instead of `pip` to install TensorFlow (one of us experienced this on Apple M2MAX).

### Using Virtual Environment
If Docker or Conda options do not work for you, set up your own virtual environment with the requirements listed in `requirements.txt`.

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Usage

### 1. Running the Server
From inside the AgnosticAPI directory, move to the `server/` directory and run:

```bash
python main.py
```

This command will launch the API, which should, by default, be listening on port 8000.

## 2. Setting up the Client

### Download the Required File
1. Download the file `15_NRR-ID27468` from the `RSE_API` folder.

### Start the Client GUI
2. Inside the AgnosticAPI directory, run:

    ```bash
    python app/client/graphical.py
    ```

    This will launch the graphical interface where you can specify important information.

    - First, update the IP address and port that the server is listening on. This information can be found on the line

 ‘Uvicorn running on …’.
    - Specify the path to `15_NRR-ID27468` and hit “start segmentation”. On the server end, you should see a message indicating that the segmentation process has started and output messages indicating the cropping process.

### 3. Verify the Returned Values/Labels
3. After the segmentation process is complete, the client should receive a “Segmentation completed!” message.
4. On the client’s system, you should see the following file: `app/Predictions/mask.npy`
5. To inspect some of the frames, run the Jupyter notebook file `AgnosticAPI/view_mask.ipynb`.

### Endpoints

- **/cv (POST)**: Performs image classification on an uploaded file.
- **/seg3d (POST)**: Performs 3D segmentation on an uploaded file.

### Data Format

#### /cv (POST)

- **Request**:
  - Accepts a file upload using the multipart/form-data format.
  - The uploaded file should be an image file (e.g., jpg, png).
  - Example:
    ```bash
    curl -X POST "http://localhost:8000/cv" -F "file=@path_to_image_file.jpg"
    ```

- **Response**:
  - Returns a JSON object with the predicted class and probability.
  - Example:
    ```json
    {
      "class": 1,
      "probability": 0.95
    }
    ```

#### /seg3d (POST)

- **Request**:
  - Accepts a file upload using the multipart/form-data format.
  - The uploaded file should be 3D medical data in a format supported by the library used (e.g., Nifti).
  - Example:
    ```bash
    curl -X POST "http://localhost:8000/seg3d" -F "uploaded_file=@path_to_nifti_file.nii" -H "uuid: your-uuid-here"
    ```

- **Response**:
  - Returns a raw byte stream containing the segmentation labels. The data type is uint8.
  - Example:
    ```json
    {
      "segmentation_labels": "base64_encoded_byte_stream"
    }
    ```

### Error Handling

Both endpoints return JSON responses with an "error" field containing a message string in case of any exceptions.

- Example:
  ```json
  {
    "error": "File not found."
  }
  ```

### Problems/Limitations

- **/seg3d (POST)**:
  - Tiling the uploaded file and processing each tile one at a time results in a long runtime. Explore ways to use GPU/parallelize.
  - The request needs file uploads (*.nii files) that are ~70 Mb. These are too large for Postman (5Mb limit). This file size also poses a problem for Swagger.

### Loading Models Using Classes

To load models using the defined classes (`CVModel` and `Seg3DModel`), follow these steps:

1. **Define the Model Class**: Create a class that extends the `Model` base class for specific model types.
2. **Initialize the Model**: Create an instance of the model class and provide the necessary parameters.
3. **Load the Model**: Use the `load` method of the model instance to load the pre-trained model from the specified path.

**Example Code for Loading Models**:

```python
from models import CVModel, Seg3DModel

# Define models by creating an instance of the model class
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

```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [GNU General Public License](LICENSE).

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/YourUsername/AgnosticAPI/workflows/CI/badge.svg
[actions-link]:             https://github.com/YourUsername/AgnosticAPI/actions
[pypi-link]:                https://pypi.org/project/agnosticapi/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/agnosticapi
[pypi-version]:             https://img.shields.io/pypi/v/agnosticapi
<!-- prettier-ignore-end -->

This README provides a comprehensive overview of your project, including installation instructions, usage examples, directory structure, endpoints, error handling, and client setup, in a standard AI model README format.
    