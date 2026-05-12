from pathlib import Path

import torch


def save_model(
        model: torch.nn.Module,
        artifact_dir: Path | str,
        model_name: str,
) -> None:
    # Create target directory
    target_dir_path = Path(artifact_dir)
    target_dir_path.mkdir(parents=True,
                            exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt")
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
                f=model_save_path)