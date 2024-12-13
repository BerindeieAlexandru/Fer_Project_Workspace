import os
import shutil

# Paths to your directories
base_dir = "fer2013_original"
test_dir = os.path.join(base_dir, "test")
validation_dir = os.path.join(base_dir, "val")

# Classes (same as the subfolder names)
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Create the val directory structure if it doesn't exist
if not os.path.exists(validation_dir):
    os.makedirs(validation_dir)
    for class_name in classes:
        os.makedirs(os.path.join(validation_dir, class_name))

# Function to move PublicTest images to val directory
def move_public_test_images():
    for class_name in classes:
        test_class_dir = os.path.join(test_dir, class_name)
        validation_class_dir = os.path.join(validation_dir, class_name)

        # Check if the test directory exists
        if os.path.exists(test_class_dir):
            # Iterate over each file in the test subdirectory
            for filename in os.listdir(test_class_dir):
                if filename.startswith("PrivateTest_"):
                    # Move the file to the corresponding val subdirectory
                    source = os.path.join(test_class_dir, filename)
                    destination = os.path.join(validation_class_dir, filename)
                    shutil.move(source, destination)
                    print(f"Moved {filename} from {test_class_dir} to {validation_class_dir}")

# Execute the function
move_public_test_images()