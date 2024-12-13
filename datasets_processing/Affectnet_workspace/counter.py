import os

# Define the base path where your dataset is located
base_dir = 'affectnet'

# Initialize a dictionary to store counts
image_counts = {}

# Loop through each subdirectory
for label_dir in os.listdir(base_dir):
    label_path = os.path.join(base_dir, label_dir)

    # Ensure it's a directory
    if os.path.isdir(label_path):
        # Count the number of image files in the directory
        num_images = len([file for file in os.listdir(label_path) if file.endswith(('.jpg', '.jpeg', '.png'))])
        image_counts[label_dir] = num_images

# Print the results
for label, count in image_counts.items():
    print(f"{label}: {count} images")

# Optional: Print total number of images
total_images = sum(image_counts.values())
print(f"\nTotal images in the dataset: {total_images}")
