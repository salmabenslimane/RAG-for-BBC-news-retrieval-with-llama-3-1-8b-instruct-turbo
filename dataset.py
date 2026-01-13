from utils import read_dataframe
import kagglehub
import os

# Make a folder in your project to store datasets
DATA_FOLDER = os.path.join(os.getcwd(), "datasets")
os.makedirs(DATA_FOLDER, exist_ok=True)

# Download the latest dataset into your project folder
path = kagglehub.dataset_download(
    "dylanjcastillo/news-headlines-2024",
    download_path=DATA_FOLDER  # <-- this ensures it's saved here
)

print("Path to dataset files:", path)

# Read the CSV directly from your project folder
DATASET = read_dataframe(f"{path}/news_data_dedup.csv")

# Optional: preview
print(DATASET[:5])
