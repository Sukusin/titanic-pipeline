import kagglehub


# Download latest version
path = kagglehub.competition_download('titanic', output_dir="deep-nn/datasets/raw")

print("Path to competition files:", path)