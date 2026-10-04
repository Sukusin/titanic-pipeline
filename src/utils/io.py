import json
import shutil
from datetime import datetime
from pathlib import Path


def save_json(data: dict, path: str | Path) -> None:
    """Write indented JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(obj=data, fp=f, indent=2)

def copy_file(src: str | Path, dst: str | Path) -> None:
    """Copy a file after creating its destination directory."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src=src, dst=dst)
    
def make_run_name(config: dict, dataset_type: str) -> str:
    """Combine experiment name, dataset type, and current timestamp."""
    return (
        f"{config['experiment']['name']}_"
        f"{dataset_type}_"
        f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    )

def make_run_dirs(run_name: str, model_type: str) -> tuple[Path, Path]:
    """Create matching log and model artifact directories."""
    run_dir = Path("logs")/model_type/run_name
    artifact_dir = Path("models")/model_type/run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, artifact_dir