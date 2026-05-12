import json
import shutil
from datetime import datetime
from pathlib import Path


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(obj=data, fp=f, indent=2)

def copy_file(src: str | Path, dst: str | Path) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src=src, dst=dst)
    
def make_run_name(config: dict, dataset_type: str) -> str:
    return (
        f"{config['experiment']['name']}_"
        f"{dataset_type}_"
        f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    )

def make_run_dirs(run_name: str, dataset_type: str) -> tuple[Path, Path]:
    run_dir = Path("logs")/dataset_type/run_name
    artifact_dir = Path("models")/dataset_type/run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, artifact_dir