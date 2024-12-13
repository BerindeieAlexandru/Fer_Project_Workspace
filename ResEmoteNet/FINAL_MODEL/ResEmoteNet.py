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
    print(f"Using {device} device")

    # Transform the dataset
    train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Transform the dataset
    eval = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Training Data
    fer_dataset_train = Four4All(csv_file='fer2013/trainb_labels.csv', img_dir='fer2013/train', transform=train)
    data_train_loader = DataLoader(fer_dataset_train, batch_size=16, shuffle=True, num_workers=4)
    train_image, train_label = next(iter(data_train_loader))

    # Validation Data
    val_dataset = Four4All(csv_file='fer2013/valb_labels.csv', img_dir='fer2013/val', transform=eval)
    data_val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    val_image, val_label = next(iter(data_val_loader))

    # Testing Data
    fer_dataset_test = Four4All(csv_file='fer2013/testb_labels.csv', img_dir='fer2013/test', transform=eval)
    data_test_loader = DataLoader(fer_dataset_test, batch_size=16, shuffle=False, num_workers=4)
    test_image, test_label = next(iter(data_test_loader))

    print(f"Train batch: Image shape {train_image.shape}, Label shape {train_label.shape}")
    print(f"Validation batch: Image shape {val_image.shape}, Label shape {val_label.shape}")
    print(f"Test batch: Image shape {test_image.shape}, Label shape {test_label.shape}")

    # # FER2013 weights
    # class_samples = [8990, 4001, 6077, 4953, 885, 5121, 6198]
    # num_classes = len(class_samples)
    # total_samples = sum(class_samples)

    # # Compute the weights for each class
    # class_weights = []
    # for count in class_samples:
    #     weight = 1 - (count / total_samples)
    #     class_weights.append(weight)

    class_weights = [0.5684, 1.2934, 0.8491, 1.0266, 9.4066, 1.0010, 0.8260]

    # Convert to a tensor and move to the appropriate device
    fer_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(fer_weights)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    
    # Load the model
    model = FourforAll()

    model.to(device)

    # Print the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')

    # Hyperparameters
    criterion = torch.nn.CrossEntropyLoss(weight=fer_weights)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.6, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, verbose=True)

    patience = 6
    best_val_acc = 0
    patience_counter = 0
    epoch_counter = 0

    num_epochs = 150

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    num_classes = 7
    class_accuracies = torch.zeros(num_classes).to(device)

    # Start training
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total, correct = 0, 0

        for data in tqdm(data_train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(data_train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation Starts
        model.eval()
        val_running_loss = 0.0
        val_targets = []
        val_predictions = []
        with torch.no_grad():
            for data in tqdm(data_val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_targets.extend(labels.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())

        val_loss = val_running_loss / len(data_val_loader)
        val_losses.append(val_loss)
        val_acc = accuracy_score(val_targets, val_predictions)
        val_accuracies.append(val_acc)

        # Print metrics
        print(f"\nEpoch {epoch+1}: ")
        print(f"\nTrain Loss: {train_loss}")
        print(f"\nTrain Accuracy: {train_acc}")
        print(f"\nValidation Loss: {val_loss}")
        print(f"\nValidation Accuracy: {val_acc}")
        print("\nValidation Classification Report:")
        print(classification_report(val_targets, val_predictions, zero_division=0))

        # Adjust learning rate based on validation accuracy
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nLearning Rate: {current_lr}")
        
        epoch_counter += 1
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0 
            torch.save({
                'model_state_dict': model.state_dict(),
            }, 'fer_model.pth')
        else:
            patience_counter += 1
            print(f"No improvement in validation accuracy for {patience_counter} epoch(s).")
        
        if patience_counter > patience:
            print("Stopping early due to lack of improvement in validation accuracy.")
            break
    df = pd.DataFrame({
        'Epoch': range(1, epoch_counter+1),
        'Train Loss': train_losses,
        'Validation Loss': val_losses,
        'Train Accuracy': train_accuracies,
        'Validation Accuracy': val_accuracies,
    })
    df.to_csv('result_loss.csv', index=False)

    # Load the best model before testing
    checkpoint = torch.load('fer_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

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
    print(classification_report(test_targets, test_predictions, zero_division=0))

if __name__ == '__main__':
    main()