from agnosticapi.server.model_definitions import CVModel
from agnosticapi.server.models.cv_model import model_files

cv_model = CVModel(
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

cv_model.load(cv_model.model_files)

print(model_files)