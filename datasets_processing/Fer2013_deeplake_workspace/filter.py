import os
from retinaface import RetinaFace

def process_images_for_faces(directory):
    no_face_count = 0
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Skip non-image files
        if not (filename.endswith('.jpg') or filename.endswith('.png')):
            continue
        
        try:
            # Check for faces using RetinaFace
            detections = RetinaFace.detect_faces(filepath, detector_backend = 'retinaface', )
            
            # If no faces detected, rename the file
            if not detections:
                no_face_count += 1
                name, ext = os.path.splitext(filename)
                
                # Check if "_noface" is already in the filename
                if "_noface" not in name:
                    new_name = f"{name}_noface{ext}"
                    new_path = os.path.join(directory, new_name)
                    os.rename(filepath, new_path)
                    
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue
    
    return no_face_count

# Usage example:
directory_path = "val"
no_face_images = process_images_for_faces(directory_path)
print(f"Number of images without faces: {no_face_images}")
