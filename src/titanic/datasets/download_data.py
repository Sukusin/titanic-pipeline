from pathlib import Path

import kagglehub


def download_main_dataset(
        handle: str = 'titanic', 
        output_dir: str | Path = "data/titanic/raw"
        ) -> None:
    """Download the latest competition files into the raw data directory."""
    path = kagglehub.competition_download(handle=handle, output_dir=output_dir, force_download=True)

    print("Path to competition files:", path)
