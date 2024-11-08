import os

# Define base directories
base_dir = 'fer2013'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Function to count images in each folder
def count_images_in_folder(path):
    return len([file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))])

# Dictionary to store image counts
image_counts = {
    'train': {},
    'val': {},
    'test': {}
}

# Count images in each set
for dataset, path in zip(['train', 'val', 'test'], [train_dir, validation_dir, test_dir]):
    total_images = 0
    for emotion in os.listdir(path):
        emotion_path = os.path.join(path, emotion)
        count = count_images_in_folder(emotion_path)
        image_counts[dataset][emotion] = count
        total_images += count
    image_counts[dataset]['total'] = total_images

# Print the results in an easy-to-see format
print("Image Counts per Folder:\n")
for dataset, counts in image_counts.items():
    print(f"{dataset.capitalize()} Set:")
    for emotion, count in counts.items():
        print(f"  {emotion.capitalize()}: {count}")
    print()

# Calculate the grand total across all sets
grand_total = image_counts['train']['total'] + image_counts['val']['total'] + image_counts['test']['total']
print(f"Grand Total: {grand_total} images")
