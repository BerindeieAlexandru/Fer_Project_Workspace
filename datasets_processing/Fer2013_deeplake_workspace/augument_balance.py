import os
import random
from collections import defaultdict
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2
import torch

# Define paths
SOURCE_DIR = "fer2013_o_f_bal/train"
DEST_DIR = "fer2013_o_f_bal/train"
TARGET_SAMPLES = 6000

# Define emotions and augmentation transforms
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Data augmentation pipeline
augmentation_transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.RandomGrayscale(p=0.2),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    v2.ToPILImage()
])

# Ensure the destination directory exists
if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

# Organize images by emotion
emotion_files = defaultdict(list)
max_indices = defaultdict(int)

for filename in os.listdir(SOURCE_DIR):
    if filename.endswith(".jpg"):
        parts = filename.split("_")
        if len(parts) == 3 and parts[2].replace(".jpg", "") in EMOTIONS:
            emotion = parts[2].replace(".jpg", "")
            index = int(parts[1])
            emotion_files[emotion].append(filename)
            max_indices[emotion] = max(max_indices[emotion], index)

# Balance the dataset
for emotion in EMOTIONS:
    files = emotion_files[emotion]
    num_files = len(files)
    print(f"Processing '{emotion}': {num_files} files found.")

    # Check if the number of images exceeds TARGET_SAMPLES
    if num_files > TARGET_SAMPLES:
        print(f"More than {TARGET_SAMPLES} images for '{emotion}'. Removing excess...")
        
        # Randomly shuffle the images and remove the excess
        excess_images = files[TARGET_SAMPLES:]  # Get images beyond the target
        for excess_image in excess_images:
            file_path = os.path.join(SOURCE_DIR, excess_image)
            os.remove(file_path)  # Delete the excess image
        
        # Keep only the first 6000 images
        files = files[:TARGET_SAMPLES]
    
    # Compute how many images to augment (if needed)
    to_augment = TARGET_SAMPLES - len(files)
    current_index = max_indices[emotion] + 1
    
    # Augment images to balance
    if to_augment > 0:
        print(f"Augmenting {to_augment} images for '{emotion}'...")
        for i in range(to_augment):
            # Randomly choose an image to augment
            src_file = random.choice(files)
            src_path = os.path.join(SOURCE_DIR, src_file)
            image = Image.open(src_path)

            # Apply augmentation
            augmented_image = augmentation_transforms(image)

            # Save augmented image with new name
            new_filename = f"train_{current_index}_{emotion}.jpg"
            new_filepath = os.path.join(DEST_DIR, new_filename)
            augmented_image.save(new_filepath)
            current_index += 1

print("Dataset balancing complete. All emotions now have 6000 samples.")
