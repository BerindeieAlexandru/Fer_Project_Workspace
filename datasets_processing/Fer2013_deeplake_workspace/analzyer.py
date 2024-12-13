import os
from collections import Counter

def count_emotions_in_directory(directory):
    emotion_counter = Counter()

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Skip non-image files
        if not filename.endswith('.jpg'):
            continue
        
        # Parse the emotion from the filename (e.g., test_0_angry.jpg)
        parts = filename.split('_')
        if len(parts) >= 3:
            try:
                # Extract the emotion name (e.g., "angry" from "test_0_angry.jpg")
                emotion_name = parts[2].split('.')[0]  # Remove extension if present
                emotion_counter[emotion_name] += 1
            except IndexError:
                print(f"Invalid filename format: {filename}")
    
    return dict(emotion_counter)


def process_fer2013_o_f(base_directory):

    subdirs = ['train', 'test', 'val']
    results = {}

    # Process each subdirectory
    for subdir in subdirs:
        subdir_path = os.path.join(base_directory, subdir)
        if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
            print(f"Processing directory: {subdir}")
            results[subdir] = count_emotions_in_directory(subdir_path)
        else:
            print(f"Directory does not exist: {subdir_path}")

    # Display results
    for subdir, counts in results.items():
        print(f"\nEmotion counts in {subdir} directory:")
        for emotion, count in counts.items():
            print(f"  {emotion.capitalize()}: {count}")


# Usage
base_directory = "fer2013_o_f_bal"
process_fer2013_o_f(base_directory)
