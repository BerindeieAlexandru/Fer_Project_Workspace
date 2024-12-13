import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
import os
from approach_mnv2 import CustomModel

def evaluate_model(loader, checkpoint_path):
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=False))
    else:
        print(f"Checkpoint {checkpoint_path} not found.")
        return

    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    # Evaluate
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

    # Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=loader.dataset.classes))

    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = CustomModel(num_classes=7, dropout_rate=0.2)
model_checkpoint = r'best_test_model.pth'

data_dir = "fer2013"
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

val_dataset = ImageFolder(root=val_dir, transform=transform_eval)
test_dataset = ImageFolder(root=test_dir, transform=transform_eval)

val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

evaluate_model(test_loader, model_checkpoint)