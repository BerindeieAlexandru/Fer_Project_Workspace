import os

def remove_noface_suffix(directory):
    renamed_count = 0
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Skip non-files
        if not os.path.isfile(filepath):
            continue
        
        # Check if '_noface' is in the filename
        if "_noface" in filename:
            name, ext = os.path.splitext(filename)
            new_name = name.replace("_noface", "")
            new_path = os.path.join(directory, f"{new_name}{ext}")
            
            # Rename the file
            os.rename(filepath, new_path)
            renamed_count += 1
    
    return renamed_count

# Usage example:
directory_path = "//wsl.localhost/Ubuntu-24.04/home/alex/TensorProjects/fer2013/train"
renamed_files = remove_noface_suffix(directory_path)
print(f"Number of files renamed: {renamed_files}")