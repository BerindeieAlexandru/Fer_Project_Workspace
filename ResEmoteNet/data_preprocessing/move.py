import os
import shutil

# Define the source directory (current structure) and destination directory (flattened structure)
source_dir = 'fer2013'  # Directory with current structured format
destination_dir = 'fer2013_ready'  # New base directory for flattened structure

# Define dataset splits
dataset_splits = ['train', 'val', 'test']

# Create the destination directory structure with flat folders for each split
for split in dataset_splits:
    os.makedirs(os.path.join(destination_dir, split), exist_ok=True)


# Function to copy images from emotion folders to a flat directory
def copy_images_to_flat_structure(split):
    # Define paths for the source split directory (e.g., rafdb/train/angry)
    # and destination flat directory (e.g., fer2013_ready/train)
    split_src_dir = os.path.join(source_dir, split)
    split_dst_dir = os.path.join(destination_dir, split)

    # Loop over each emotion subdirectory within the split
    for emotion in os.listdir(split_src_dir):
        emotion_path = os.path.join(split_src_dir, emotion)

        if os.path.isdir(emotion_path):  # Make sure it's a directory
            # Copy each image from the current emotion folder to the flat destination folder
            for img_name in os.listdir(emotion_path):
                src = os.path.join(emotion_path, img_name)
                dst = os.path.join(split_dst_dir, img_name)  # Keep original filename

                # Copy the image to the flat structure
                shutil.copy2(src, dst)


# Execute the copy operation for each dataset split
for split in dataset_splits:
    copy_images_to_flat_structure(split)

print("Images successfully copied to the new directory structure in 'fer2013_ready'.")
