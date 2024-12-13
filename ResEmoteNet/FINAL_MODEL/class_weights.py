import os
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Define the base directory where your dataset is located
base_dir = r"D:\Alex\Desktop\final_datasets\fer2013"

# Directory to analyze
train_dir = 'train'

# Dictionary to store emotion counts for the train directory
emotion_counts = defaultdict(int)

# List to collect all emotion labels
emotion_labels = []

# Path to the train directory
train_path = os.path.join(base_dir, train_dir)

# Loop through each image file in the train directory
if os.path.exists(train_path):
    for file in os.listdir(train_path):
        # Ensure it's an image file
        if file.endswith(('.jpg', '.jpeg', '.png')):
            # Extract emotion from the file name (assumes format: purpose_id_emotion.png)
            parts = file.split('_')
            if len(parts) == 3:  # Ensure correct format
                emotion = parts[2].split('.')[0]  # Extract emotion without extension
                emotion_counts[emotion] += 1
                emotion_labels.append(emotion)  # Add emotion to the list

# Print emotion counts
print("Emotion Counts in Train Directory:")
for emotion, count in emotion_counts.items():
    print(f"  {emotion}: {count} images")

# Calculate class weights
unique_emotions = list(emotion_counts.keys())
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array(unique_emotions),
    y=np.array(emotion_labels)
)

# Create a dictionary mapping emotions to weights
class_weight_dict = {emotion: weight for emotion, weight in zip(unique_emotions, class_weights)}

# Print the class weights
print("\nClass Weights for Train Dataset:")
for emotion, weight in class_weight_dict.items():
    print(f"  {emotion}: {weight:.4f}")
