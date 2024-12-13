import os
import random

# Define the base path to the validation directory
val_dir = 'D:/Alex/Desktop/datasets_processing/fer+semibalanced_workspace/fer_plus/val'

# Target number of images per emotion
target_count = 508

# Define the list of emotions
emotions = ['angry', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']

# Dictionary to track images by emotion
emotion_images = {emotion: [] for emotion in emotions}

# Iterate through all files in the validation directory
for file in os.listdir(val_dir):
    if file.endswith(('.jpg', '.jpeg', '.png')):
        # Extract the emotion from the filename (assumes format: val_id_emotion.png)
        parts = file.split('_')
        if len(parts) == 3:  # Ensure correct format
            emotion = parts[2].split('.')[0]
            if emotion in emotions:
                emotion_images[emotion].append(file)

# Balance each emotion class
for emotion, files in emotion_images.items():
    if len(files) > target_count:
        # Calculate how many images to remove
        surplus_count = len(files) - target_count
        
        # Randomly select images to remove
        images_to_remove = random.sample(files, surplus_count)
        
        # Remove the selected images
        for image in images_to_remove:
            image_path = os.path.join(val_dir, image)
            os.remove(image_path)
            print(f"Removed {image}")
    
    # Print the final count for each emotion
    remaining_images = len([f for f in os.listdir(val_dir) if f.endswith(emotion + '.png') or f.endswith(emotion + '.jpg')])
    print(f"{emotion.capitalize()}: {remaining_images} images after balancing.")

print("Balancing complete!")
