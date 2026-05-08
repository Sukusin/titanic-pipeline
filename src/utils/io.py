import json
import shutil
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
    