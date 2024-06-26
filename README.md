# AgnosticAPI

Directory tree with annotations:
```
AGNOSTICAPI
│
├── .ipynb_checkpoints     # (1) Jupyter notebook checkpoints for saving states during development.
│
├── app                    # (2) Main application directory
│   ├── client             # (3) Client-side code for the application
│   │   ├── __pycache__    # (4) Cached bytecode files for faster execution.
│   │   ├── Predictions    # (5) Directory to store prediction results.
│   │   │   └── mask.npy   # (6) Numpy file containing the prediction mask.
│   │   ├── graphical.py   # (7) Script for the GUI client using tkinter for file selection and segmentation.
│   │   └── segmentation_client.py   # (8) Script to handle the client-side segmentation requests.
│   │
│   ├── server             # (9) Server-side code for the application.
│       ├── models         # (10) Directory to store the ML models.
│       │   ├── cv_model   # (11) Computer vision model directory.
│       │   │   └── MobileNetv2_model.keras   # (12) MobileNetV2 model file for image classification.
│       │   ├── ls_seg3d_model   # (13) Directory for 3D segmentation model.
│       │   ├── joe_download_and_run.py   # (14) Script for downloading and running Joe's model.
│       │   └── model.py   # (15) Script for handling model loading and inference.
│       │
│       ├── seg3d_backend   # (16) Backend for handling 3D segmentation.
│           ├── __pycache__   # (17) Cached bytecode files for faster execution.
│           ├── ls_seg3d_utils   # (18) Utilities for 3D segmentation.
│           ├── c3d_linux   # (19) Compiled 3D segmentation binaries for Linux.
│           ├── c3d_macos_arm   # (20) Compiled 3D segmentation binaries for MacOS ARM.
│           ├── ls_seg3d.py   # (21) Main script for 3D segmentation.
│       └── main.py   # (22) Main script for starting the FastAPI server.
│
├── media   # (23) Directory for storing media files (empty here).
│
├── .gitignore   # (24) Git ignore file to exclude files from version control.
├── 15_NRR-ID27468.nii   # (25) Sample NIfTI file for testing.
├── Dockerfile   # (26) Docker configuration file for containerizing the application.
├── EcoVizAPI_seg3d_joe_contract.docx   # (27) Documentation related to the EcoViz API contract.
├── README.md   # (28) Readme file containing the project overview and instructions.
├── requirements.txt   # (29) List of Python dependencies for the project.
├── seg3Denv.yml   # (30) Conda environment configuration file.
├── seg3DenvMini.yml   # (31) Minimal Conda environment configuration file.
├── test.txt   # (32) Test file (purpose unclear from the name).
└── view_mask.ipynb   # (33) Jupyter notebook for visualizing prediction masks.

```


To run, you will need:
1. **Docker** https://www.docker.com/products/docker-desktop
    Optional: Create DockerHub account to save and manage your images and containers.
2. **Postman** https://www.postman.com/downloads/
    This will allow you to send and receive requests with the API. You will need to create an account through your institution. 

Navigate to your AgnosticAPI directory and run the code below in your terminal:

```
docker build -t mobile-net-app .
docker run -p 80:80 mobile-net-app
```

1. Copy URL output from Terminal: ![alt text](media/image.png)
2. In postman, create a new collection: ![alt text](media/image2.png)
3. Add a request, change to a POST request: ![alt text](media/image4.png)
4. Navigate to Body tab: ![alt text](media/image3.png)
5. Paste your http address in the field next to POST and add `/predict`: ![alt text](media/image5.png) This calls the prediction function in the main.py file.
6. Change dropdown menu from `Text ` to `File` and then upload a JPEG file from your local directory. 

Then in postman, set the url to: http://localhost:80

In postman, create a new POST request
set the url to: http://localhost:80/predict
Click on the Body tab
Select form-data as the body type
Add a key named file (matches the parameter name in the FastAPI function)
Click Browse and select the image file you want to classify
Click the Send button to send the POST request with the image
