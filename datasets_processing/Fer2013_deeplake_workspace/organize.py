import os
import shutil

# Define source and target directories
source_dir = "fer2013_emonext"
target_dir = "fer2013"

# Define the emotion labels
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Create the target directory structure
for split in ["train", "test", "val"]:
    for emotion in emotions:
        os.makedirs(os.path.join(target_dir, split, emotion), exist_ok=True)

# Function to organize images by emotion
def organize_images():
    for split in ["train", "test", "val"]:
        split_dir = os.path.join(source_dir, split)
        for image_name in os.listdir(split_dir):
            # Parse the emotion from the filename
            try:
                _, _, emotion = image_name.split('_')
                emotion = emotion.split('.')[0]  # Remove file extension

                if emotion in emotions:
                    # Source and target paths
                    source_path = os.path.join(split_dir, image_name)
                    target_path = os.path.join(target_dir, split, emotion, image_name)

                    # Move the file
                    shutil.copy2(source_path, target_path)
            except ValueError:
                print(f"Skipping file with unexpected format: {image_name}")

# Run the organization process
organize_images()
print("Dataset reorganization complete.")
