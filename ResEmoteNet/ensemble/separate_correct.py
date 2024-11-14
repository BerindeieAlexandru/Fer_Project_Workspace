import os
import shutil
from sklearn.model_selection import train_test_split

# Define the paths
original_train_dir = r"D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\ensemble\fer2013_0\train"
new_train_dir = r"D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\ensemble\fer2013_0\trainnew"
new_val_dir = r"D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\ensemble\fer2013_0\validation"

# Make new directories for train and validation if they don't exist
os.makedirs(new_train_dir, exist_ok=True)
os.makedirs(new_val_dir, exist_ok=True)

# Get list of classes from the original train directory
classes = os.listdir(original_train_dir)

# Iterate over each class
for class_name in classes:
    class_dir = os.path.join(original_train_dir, class_name)

    if os.path.isdir(class_dir):
        # Create class subdirectories in new_train_dir and new_val_dir
        os.makedirs(os.path.join(new_train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(new_val_dir, class_name), exist_ok=True)

        # Get all images in the class directory
        image_files = os.listdir(class_dir)
        image_paths = [os.path.join(class_dir, img) for img in image_files]

        # Split images into train and validation (80% train, 20% validation)
        train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

        # Move files to new train directory
        for train_path in train_paths:
            shutil.copy(train_path, os.path.join(new_train_dir, class_name))

        # Move files to new validation directory
        for val_path in val_paths:
            shutil.copy(val_path, os.path.join(new_val_dir, class_name))

print("Data splitting complete!")
