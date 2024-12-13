import os

def count_samples(directory):
    class_counts = {}
    for root, dirs, files in os.walk(directory):
        for sub_dir in dirs:
            class_path = os.path.join(root, sub_dir)
            class_name = os.path.basename(class_path)
            file_count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
            class_counts[class_name] = file_count
    return class_counts

# Paths to the train and test directories
train_dir = "fer2013_original/train"
test_dir = "fer2013_original/test"
val_dir = "fer2013_original/val"

# Count samples
train_counts = count_samples(train_dir)
test_counts = count_samples(test_dir)
val_counts = count_samples(val_dir)

# Print results
print("Train Class Counts:")
for class_name, count in train_counts.items():
    print(f"{class_name}: {count}")

print("\nValidation Class Counts:")
for class_name, count in val_counts.items():
    print(f"{class_name}: {count}")

print("\nTest Class Counts:")
for class_name, count in test_counts.items():
    print(f"{class_name}: {count}")
