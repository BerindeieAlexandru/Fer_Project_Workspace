import os

# Define base paths
base_dir = 'fer2013_original'  # Original directory
new_base_dir = 'fer2013_o'  # New base directory for renamed files
dataset_dirs = {'train': 'train', 'validation': 'val', 'test': 'test'}  # Mapping original to new names

# Create new directory structure
for original_split, new_split in dataset_dirs.items():
    split_path = os.path.join(new_base_dir, new_split)
    os.makedirs(split_path, exist_ok=True)
    # Only create subdirectories if the original directory exists (handles validation correctly)
    original_split_path = os.path.join(base_dir, original_split)
    if os.path.exists(original_split_path):
        for emotion in os.listdir(original_split_path):
            os.makedirs(os.path.join(split_path, emotion), exist_ok=True)


# Function to rename and move images
def rename_images_in_folder(split, emotion, src_folder, dst_folder):
    # List all images in the current emotion folder
    images = [img for img in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, img))]

    for idx, img_name in enumerate(images, start=1):
        # Create new filename in the format: <split>_<index>_<emotion>.jpg
        new_name = f"{split}_{idx}_{emotion}.jpg"

        # Define source and destination paths
        src = os.path.join(src_folder, img_name)
        dst = os.path.join(dst_folder, new_name)

        # Rename and move the file
        os.rename(src, dst)


# Rename images in each dataset split
for original_split, new_split in dataset_dirs.items():
    split_path = os.path.join(base_dir, original_split)
    new_split_path = os.path.join(new_base_dir, new_split)

    if os.path.exists(split_path):  # Ensure the original directory exists
        for emotion in os.listdir(split_path):
            emotion_src_path = os.path.join(split_path, emotion)
            emotion_dst_path = os.path.join(new_split_path, emotion)

            # Rename and move images in each emotion folder
            rename_images_in_folder(new_split, emotion, emotion_src_path, emotion_dst_path)

print("Images successfully renamed and moved to the new directory structure.")
