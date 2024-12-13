import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import Dataset
from dataset_loader import Four4All
from arhitecture import FourforAll
from sklearn.metrics import classification_report, accuracy_score


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FourforAll()
    model.to(device)

    # Load the best model before testing
    checkpoint = torch.load(r"fer_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    fer_dataset_test = Four4All(csv_file='fer/test_o_labels.csv', img_dir='fer/test_o', transform=transform)
    data_test_loader = DataLoader(fer_dataset_test, batch_size=16, shuffle=False)
    test_image, test_label = next(iter(data_test_loader))

    # FER2013 weights
    class_samples = [8990, 4001, 6077, 4953, 885, 5121, 6198]
    num_classes = len(class_samples)
    total_samples = sum(class_samples)

    # Compute the weights for each class
    class_weights = []
    for count in class_samples:
        weight = 1 - (count / total_samples)
        class_weights.append(weight)


    # Convert to a tensor and move to the appropriate device
    fer_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=fer_weights)

    # Final test with the best model
    model.eval()
    test_targets = []
    test_predictions = []
    test_running_loss = 0.0
    with torch.no_grad():
        for data in tqdm(data_test_loader, desc="Testing with Best Model"):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_targets.extend(labels.cpu().numpy())
            test_predictions.extend(predicted.cpu().numpy())

    test_loss = test_running_loss / len(data_test_loader)
    test_acc = accuracy_score(test_targets, test_predictions)


    print(f"Final Test Loss: {test_loss}")
    print(f"Final Test Accuracy: {test_acc}")
    print("Test Classification Report:")
    print(classification_report(test_targets, test_predictions))

if __name__ == '__main__':
    main()