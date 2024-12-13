import os
from PIL import Image

# Define the base path to the AffectNet dataset
affectnet_dir = 'affectnet'  # Replace with your actual AffectNet directory path

# Define the target size for resizing
target_size = (64, 64)

# List of emotions in AffectNet
emotions = ['angry', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']

# Function to process and save images
def process_image(image_path, output_path):
    try:
        # Open the image file
        with Image.open(image_path) as img:
            # Convert the image to grayscale
            img_gray = img.convert('L')
            # Resize the image to 64x64
            img_resized = img_gray.resize(target_size)
            # Save the processed image
            img_resized.save(output_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Process images in each emotion directory
for emotion in emotions:
    emotion_dir = os.path.join(affectnet_dir, emotion)
    
    # Check if the directory exists
    if os.path.exists(emotion_dir):
        # Iterate through each image in the emotion folder
        for image_name in os.listdir(emotion_dir):
            if image_name.endswith(('.jpg', '.jpeg', '.png')):
                # Full path to the image
                image_path = os.path.join(emotion_dir, image_name)
                # Define the output path (overwrite original or save in a new directory)
                output_path = image_path  # You can change this to save in another folder if needed

                # Process the image
                process_image(image_path, output_path)

                print(f"Processed and saved {image_name} in {emotion} directory.")

print("Image processing complete!")
