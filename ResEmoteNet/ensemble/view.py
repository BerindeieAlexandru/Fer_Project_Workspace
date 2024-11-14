import os

# Paths to your directories
base_dir = "fer2013_o"
categories = ["train", "test", "validation"]
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# Function to count samples in each category and class
def count_samples():
    for category in categories:
        category_dir = os.path.join(base_dir, category)
        print(f"\nCategory: {category}")

        total_samples = 0

        for class_name in classes:
            class_dir = os.path.join(category_dir, class_name)

            if os.path.exists(class_dir):
                # Count number of files in the directory
                num_samples = len(os.listdir(class_dir))
                total_samples += num_samples
                print(f"  {class_name}: {num_samples} samples")
            else:
                print(f"  {class_name}: 0 samples (directory not found)")

        # Display total count for the category
        print(f"Total samples in {category}: {total_samples}")


# Execute the function
count_samples()
