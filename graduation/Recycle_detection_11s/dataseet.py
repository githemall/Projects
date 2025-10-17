import kagglehub

download_dir = "/home/asd/projects/Portfolio/graduation/Recycle_detection11/taco_dataset"

# Download latest version
path = kagglehub.dataset_download("vencerlanz09/taco-dataset-yolo-format")

print("Path to dataset files:", path)