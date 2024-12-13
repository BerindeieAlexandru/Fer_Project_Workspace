import os
import deeplake
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image

# Load the dataset
# change to fer2013-train, fer2013-public-test, fer2013-private-test
ds = deeplake.load('hub://activeloop/fer2013-train')

# Define the mapping from label indices to emotion names
emotion_classes = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

# Create a directory to store the images
# change to train, test, val
output_dir = "train"
os.makedirs(output_dir, exist_ok=True)

# Initialize counters for each emotion to track the appearance index
appearance_counter = defaultdict(int)

# Iterate through the dataset and save images
for sample in ds:
    # Extract the image and label
    image = sample['images'].numpy()
    label = int(sample['labels'].numpy().item())  # Extract scalar value from label tensor
    
    # Get the emotion name
    emotion = emotion_classes[label]
    
    # Generate the filename
    appearance_index = appearance_counter[label]
    # change to train, test, val
    filename = f"train_{appearance_index}_{emotion}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    # Save the image using PIL
    img = Image.fromarray(image)  # Convert NumPy array to PIL Image
    img = img.convert("L")  # Convert to grayscale
    img.save(filepath)
    
    # Increment the appearance counter for this label
    appearance_counter[label] += 1

print(f"All images saved in the '{output_dir}' directory!")
