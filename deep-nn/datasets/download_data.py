import kagglehub
from pathlib import Path

def download_main_dataset(
        handle: str = 'titanic', 
        output_dir: str | Path = "deep-nn/datasets/raw"
        ) -> None:
    # Download latest version
    path = kagglehub.competition_download(handle=handle, output_dir=output_dir, force_download=True)

    print("Path to competition files:", path)