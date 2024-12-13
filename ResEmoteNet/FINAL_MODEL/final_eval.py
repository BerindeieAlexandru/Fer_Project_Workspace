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
import seaborn as sns
import os
from PIL import Image
from torch.utils.data import Dataset
from dataset_loader import Four4All
from arhitecture import FourforAll
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def plot_metrics(result_csv_path):
    # Load results from CSV
    results = pd.read_csv(result_csv_path)
    epochs = results['Epoch']
    train_loss = results['Train Loss']
    val_loss = results['Validation Loss']
    train_acc = results['Train Accuracy']
    val_acc = results['Validation Accuracy']

    # Plot Train vs Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label='Train Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='orange')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Train vs Validation Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, label='Train Accuracy', color='blue')
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='orange')
    plt.title('Train vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

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

    fer_dataset_test = Four4All(csv_file='fer_plus/test_labels.csv', img_dir='fer_plus/test', transform=transform)
    data_test_loader = DataLoader(fer_dataset_test, batch_size=16, shuffle=False)

    # # FER2013 weights
    # class_samples = [8990, 4001, 6077, 4953, 885, 5121, 6198]
    # num_classes = len(class_samples)
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # # Compute the weights for each class
    # total_samples = sum(class_samples)
    # class_weights = [1 - (count / total_samples) for count in class_samples]
    class_weights = [0.5352, 1.1317, 1.1484, 1.6309, 21.1257, 6.1831, 0.3914]
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

    # Plot metrics from CSV
    plot_metrics('result_loss.csv')

    # Plot Confusion Matrix
    plot_confusion_matrix(test_targets, test_predictions, class_names)

if __name__ == '__main__':
    main()
