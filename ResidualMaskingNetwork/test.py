import os
import glob
import json
import cv2
import torch
import numpy as np
from torchvision.transforms import transforms
from sklearn.metrics import classification_report, accuracy_score
from models import resmasking_dropout1
import torch.nn.functional as F
import random
import imgaug

seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
is_cuda = torch.cuda.is_available()

# Load emotion dictionary
FER_2013_EMO_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

def ensure_color(image):
    if len(image.shape) == 2:
        return np.dstack([image] * 3)
    elif image.shape[2] == 1:
        return np.dstack([image] * 3)
    return image
def ensure_gray(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        pass
    return image

# Load model
def load_model(checkpoint_path):
    model = resmasking_dropout1(in_channels=3, num_classes=7)
    if is_cuda:
        model = model.cuda()
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["net"])
    model.eval()
    return model

# Process directory
def process_directory(directory_path, model, image_size):
    accuracy = 0
    report = 0
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    y_true = []
    y_pred = []

    # Iterate over images in directory
    image_paths = glob.glob(os.path.join(directory_path, "*.png"))
    for image_path in image_paths:
        # Extract true label from filename
        file_name = os.path.basename(image_path)
        true_label = file_name.split("_")[-1].split(".")[0]
        y_true.append(true_label)

        # Load and preprocess image
        image = cv2.imread(image_path)
        image = ensure_color(image)
        image = cv2.resize(image, (224,224))

        image = transform(image)
        if is_cuda:
            image = image.cuda(0)
        image = torch.unsqueeze(image, dim=0)

        # Predict emotion
        with torch.no_grad():
            output = torch.squeeze(model(image).cpu(), 0)
            probabilities = F.softmax(output, 0)
            emo_proba, emo_idx = torch.max(probabilities, dim=0)
            predicted_idx = emo_idx.item()
            predicted_label = FER_2013_EMO_DICT[predicted_idx]
            y_pred.append(predicted_label)

    # Calculate and return metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=list(FER_2013_EMO_DICT.values()), zero_division=0)
    return accuracy, report

# Main function
def main(directory_path, checkpoint_path, config_path):
    # Load configuration
    with open(config_path, "r") as file:
        configs = json.load(file)
    image_size = (configs["image_size"], configs["image_size"])

    # Load model
    model = load_model(checkpoint_path)

    # Process images and calculate metrics
    accuracy, report = process_directory(directory_path, model, image_size)

    # Print results
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    import sys
    directory_path = sys.argv[1]  # Directory containing images
    checkpoint_path = "pretrained_ckpt"  # Update with actual path
    config_path = "./configs/fer2013_config.json"  # Update with actual path

    if os.path.exists(directory_path):
        main(directory_path, checkpoint_path, config_path)
    else:
        print(f"Directory not found: {directory_path}")
