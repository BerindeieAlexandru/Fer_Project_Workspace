import os
import shutil
from collections import defaultdict

# Paths (modify as necessary)
affectnet_dir = "D:/Alex/Desktop/datasets_processing/fer+semibalanced_workspace/affectnet"
fer_plus_val_dir = "D:/Alex/Desktop/datasets_processing/fer+semibalanced_workspace/fer_plus/train"

# Emotions
emotions = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]

# Target number of images per emotion
target_count = 3000

# Valid image extensions
valid_extensions = {".png", ".jpg", ".jpeg"}

# Step 1: Count existing images in FER+ val directory
def count_existing_images(val_dir):
    emotion_counts = defaultdict(int)
    for file in os.listdir(val_dir):
        if file.lower().endswith(tuple(valid_extensions)):
            for emotion in emotions:
                if f"_{emotion}.png" in file or f"_{emotion}.jpg" in file:
                    emotion_counts[emotion] += 1
                    break
    return emotion_counts

# Step 2: Move images to meet the target count
def move_images_to_fer_plus(affectnet_dir, fer_plus_val_dir, existing_counts):
    for emotion in emotions:
        # Calculate deficit
        deficit = target_count - existing_counts.get(emotion, 0)
        if deficit <= 0:
            print(f"{emotion} already has {existing_counts[emotion]} images. Skipping.")
            continue
        
        # Source and destination directories
        source_dir = os.path.join(affectnet_dir, emotion)
        if not os.path.exists(source_dir):
            print(f"Source directory for {emotion} not found: {source_dir}")
            continue

        # Gather images from AffectNet (support multiple extensions)
        images = [img for img in os.listdir(source_dir) if img.lower().endswith(tuple(valid_extensions))]
        if len(images) < deficit:
            print(f"Not enough images in AffectNet for {emotion}. Found {len(images)}, needed {deficit}.")
            deficit = len(images)  # Take all available if less than needed
        
        # Move images
        for i, img in enumerate(images[:deficit]):
            src_path = os.path.join(source_dir, img)
            dest_filename = f"train_{existing_counts[emotion] + i + 1}_{emotion}.png"
            dest_path = os.path.join(fer_plus_val_dir, dest_filename)
            shutil.move(src_path, dest_path)

        print(f"Moved {deficit} images for {emotion}.")

# Main Execution
if __name__ == "__main__":
    # Step 1: Count existing images in FER+ val directory
    existing_counts = count_existing_images(fer_plus_val_dir)
    print("Existing image counts:", existing_counts)

    # Step 2: Move images from AffectNet to meet target count
    move_images_to_fer_plus(affectnet_dir, fer_plus_val_dir, existing_counts)

    # Step 3: Verify final counts
    final_counts = count_existing_images(fer_plus_val_dir)
    print("Final image counts:", final_counts)
