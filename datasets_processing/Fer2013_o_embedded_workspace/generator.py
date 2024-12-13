import os
import pandas as pd
from deepface import DeepFace
import numpy as np

# Dataset directories
base_dir = "fer2013"
subdirs = ["train", "val", "test"]

# Prepare a list for image metadata
image_data = []

for subdir in subdirs:
    subdir_path = os.path.join(base_dir, subdir)
    for img_name in os.listdir(subdir_path):
        emotion = img_name.split("_")[2].split(".")[0]
        img_path = os.path.join(subdir_path, img_name)
        image_data.append({"path": img_path, "emotion": emotion, "usage": subdir})

df = pd.DataFrame(image_data)
embeddings = []
labels = []
usages = []

for idx, row in df.iterrows():
    try:
        embedding = DeepFace.represent(img_path=row['path'], detector_backend ="retinaface", align=True)
        embeddings.append(embedding[0]['embedding'])
        labels.append(row['emotion'])
        usages.append(row['usage'])
    except Exception as e:
        print(f"Error processing {row['path']}: {e}")

# Save as CSV
output_data = pd.DataFrame(embeddings)
output_data['emotion'] = labels
output_data['usage'] = usages
output_data.to_csv("fer2013_embeddings.csv", index=False)