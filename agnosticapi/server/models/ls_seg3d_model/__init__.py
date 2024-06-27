from pathlib import Path

root_dir = Path(__file__).parent

model_files = root_dir / "m20230623-163203wh500epochs"
model_history = root_dir / "h20230623-163203wh500epochs"
code_dir = root_dir / "seg3d_backend"