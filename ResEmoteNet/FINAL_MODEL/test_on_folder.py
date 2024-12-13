import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score
from arhitecture import FourforAll

def evaluate_all_classes(folder_path, class_mapping, model_path='fer_model.pth'):
    """
    Evaluates a directory of images across all emotion classes.
    
    :param folder_path: Path to the directory containing labeled images.
    :param class_mapping: A dictionary mapping emotion names to class indices.
    :param model_path: Path to the trained model file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Load the model
    model = FourforAll()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Initialize results
    true_labels = []
    predicted_labels = []

    # Iterate over images in the folder
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        # Extract the true label from the file name
        try:
            emotion_name = image_name.split('_')[-1].split('.')[0]
            if emotion_name not in class_mapping:
                print(f"Skipping file '{image_name}' with unknown emotion '{emotion_name}'")
                continue
            true_label = class_mapping[emotion_name]
        except Exception as e:
            print(f"Error extracting emotion from '{image_name}': {e}")
            continue

        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Predict with the model
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output.data, 1)

            # Append the ground truth and prediction
            true_labels.append(true_label)
            predicted_labels.append(predicted.item())

        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            continue

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, target_names=class_mapping.keys())

    print(f"Overall Accuracy: {accuracy}")
    print("Classification Report:\n")
    print(report)


if __name__ == '__main__':
    # Example class mapping for your dataset
    class_mapping = {
        'happy': 0,
        'surprised': 1,
        'sad': 2,
        'angry': 3,
        'disgusted': 4,
        'fear': 5,
        'neutral': 6
    }

    # Path to the folder with images
    folder_path = r"D:\Alex\Desktop\datasets_processing\fer+original_workspace\fer_plus\val"

    # Evaluate the folder
    evaluate_all_classes(folder_path, class_mapping)
