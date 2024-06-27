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
│       │   │   └── m20230623-163203wh500epochs   # (13a) Seg3D Tensorflow model (current version).
│       │   │   └── h20230623-163203wh500epochs   # (13b) Tensorflow model history (previous versions).
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
├── media   # (23) Directory for storing media files.
│
├── .gitignore   # (24) Git ignore file to exclude files from version control.
├── 15_NRR-ID27468.nii   # (25) Sample NIfTI file for testing.
├── Dockerfile   # (26) Docker configuration file for containerizing the application.
├── EcoVizAPI_seg3d_joe_contract.docx   # (27) EcoViz API contract with rules for engagement.
├── README.md   # (28) Readme file containing the project overview and instructions.
├── requirements.txt   # (29) List of Python dependencies for the project.
├── seg3DenvMini.yml   # (30) Minimal Conda environment configuration file.
├── test.txt   # (31) Test file (purpose unclear from the name).
└── view_mask.ipynb   # (32) Jupyter notebook for visualizing prediction masks.


```


1. **Set up your environment:** To run, you will need to set up your environment. There are three options for doing this:
    
1a. On Windows, we recommend Docker:
```
docker build -t mobile-net-app .
docker run -p 80:80 mobile-net-app
```

1b. On Mac (with Mac chip), we recommend using `conda`. From within the AgnosticAPI directory, please run (line by line):
```
conda env create -n apienv python=3.8
conda activate apienv
```
For me, this yml script was being problematic, so instead I wanted to just install the dependencies one by one, and that worked.

```
conda env update -f seg3DenvMini.yml 
```

1c. If these options do not work for you, you can set up your own virtual environment with the requirements listed in `requirements.txt`.

2. **Start the Server:**
    1. From inside the AgnosticAPI directory, move to the app/server/ directory and run:
        
        ```bash
        python main.py
        ```
        
        This command will launch the API, which should, by default, be listening on port 8000.

### Setting up the Client
3. **Download the Required File:**
    1. Download the file `15_NRR-ID27468` from the `RSE_API` folder.
    **Start the Client GUI:**
    2. Inside the AgnosticAPI directory, run:
        
        ```bash
        python app/client/graphical.py
        ```
        
        This will launch the graphical interface where you can specify important information. 
        
        First update the IP address and port that the server is listening on. This information can be found on the line ‘Uvicorn running on …’.

        Specify the path to `15_NRR-ID27468` and hit “start segmentation”. On the server end, you should see a message indicating that the segmentation process has started and output messages indicating the cropping process.
        
4. **Verify the Returned Values/Labels:**
    1. After the segmentation process is complete, the client should receive a “Segmentation completed!” message.
    2. On the client’s system, you should see the following file: `app/Predictions/mask.npy`
    3. To inspect some of the frames, run the Jupyter notebook file `AgnosticAPI/view_mask.ipynb`.