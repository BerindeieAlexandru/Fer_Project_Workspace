import os
from collections import defaultdict

# Define the base directory where your dataset is located
base_dir = "D:/Alex/Desktop/datasets_processing/fer+semibalanced_workspace/fer_plus_balanced"

# Directories to analyze
directories = ['train', 'test', 'val']

# Dictionary to store emotion counts for each directory
emotion_counts = {dir_name: defaultdict(int) for dir_name in directories}

# Loop through each directory
for dir_name in directories:
    dir_path = os.path.join(base_dir, dir_name)
    
    # Ensure the directory exists
    if os.path.exists(dir_path):
        # Loop through each image file in the directory
        for file in os.listdir(dir_path):
            # Ensure it's an image file
            if file.endswith(('.jpg', '.jpeg', '.png')):
                # Extract emotion from the file name (assumes format: purpose_id_emotion.png)
                parts = file.split('_')
                if len(parts) == 3:  # Ensure correct format
                    emotion = parts[2].split('.')[0]  # Extract emotion without extension
                    emotion_counts[dir_name][emotion] += 1

# Print the results
for dir_name, counts in emotion_counts.items():
    print(f"\n{dir_name.capitalize()} directory:")
    for emotion, count in counts.items():
        print(f"  {emotion}: {count} images")

# Optional: Print total counts for each directory
for dir_name in directories:
    total = sum(emotion_counts[dir_name].values())
    print(f"\nTotal images in {dir_name}: {total}")
