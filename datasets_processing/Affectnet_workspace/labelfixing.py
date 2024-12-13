import os
import shutil
import pandas as pd

# Define the base path where your dataset and CSV are located
base_dir = 'affectnet'
csv_file = 'affectnet/labels.csv'

# Load the CSV into a DataFrame
df = pd.read_csv(csv_file)

for _, row in df.iterrows():
    # Extract relevant information
    relative_path = row['pth']
    correct_label = row['label']


    current_dir, filename = os.path.split(relative_path)

    current_path = os.path.join(base_dir, current_dir, filename)
    target_dir = os.path.join(base_dir, correct_label)
    target_path = os.path.join(target_dir, filename)

    # Check if the file exists and the current directory doesn't match the correct label
    if os.path.exists(current_path) and current_dir != correct_label:
        # Create the target directory if it doesn't exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Move the file to the correct directory
        shutil.move(current_path, target_path)
        print(f"Moved {filename} from {current_dir} to {correct_label}")

print("Relabeling complete!")
