import os
import shutil
import random


def create_validation_split(train_dir, val_dir, split_ratio=0.2):
    # Ensure the val directory exists
    os.makedirs(val_dir, exist_ok=True)

    for class_name in os.listdir(train_dir):
        class_train_path = os.path.join(train_dir, class_name)
        class_val_path = os.path.join(val_dir, class_name)

        if os.path.isdir(class_train_path):
            # Create the same class subdirectory in the val directory
            os.makedirs(class_val_path, exist_ok=True)

            # List all files in the class directory
            files = [f for f in os.listdir(class_train_path) if os.path.isfile(os.path.join(class_train_path, f))]

            # Shuffle and select 20% of files for val
            random.shuffle(files)
            val_count = int(len(files) * split_ratio)
            val_files = files[:val_count]

            # Move selected files to the val directory
            for file_name in val_files:
                src_file = os.path.join(class_train_path, file_name)
                dest_file = os.path.join(class_val_path, file_name)
                shutil.move(src_file, dest_file)

            print(f"Moved {len(val_files)} files from {class_name} to validation set.")


# Define paths
train_dir = "fer2013_balanced/train"
val_dir = "fer2013_balanced/val"

# Split data
create_validation_split(train_dir, val_dir, split_ratio=0.2)
