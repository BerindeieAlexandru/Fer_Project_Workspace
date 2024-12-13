import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights 
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torch.nn.utils as nn_utils
import torch.optim as optim
from tqdm import tqdm
import os
from approach_mnv2 import *
from collections import Counter
import numpy as np


model = CustomModel(num_classes=7, dropout_rate=0.2)

# # Load Pretrained Weights for finetuining or continue training
# model.load_state_dict(torch.load("best_val_model.pth", weights_only=False))

# Data Augmentation and Preprocessing
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Data Using ImageFolder
data_dir = "fer2013"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

train_dataset = ImageFolder(root=train_dir, transform=transform_train)
val_dataset = ImageFolder(root=val_dir, transform=transform_eval)
test_dataset = ImageFolder(root=test_dir, transform=transform_eval)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Hyperparameters
EPOCHS = 100
patience_es = 10
patience_lr = 5
min_delta = 0.01
min_lr = 1e-6
factor = 0.1
early_stopping_counter = 0
reduce_lr_counter = 0
current_lr = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calculate class weights
class_counts = Counter([label for _, label in train_dataset.samples])
total_samples = sum(class_counts.values())
class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
weights = torch.tensor([class_weights[i] for i in range(len(class_counts))], dtype=torch.float).to(device)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_val_acc = 0
best_test_acc = 0

# Move model to device
model = model.to(device)

# Training Loop
for epoch in range(100):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        max_norm = 3.0
        nn_utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {running_loss/len(train_loader)}, Train Accuracy: {100.*correct/total}%")

    # Validation Loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = 100. * correct / total
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_acc:.2f}%")

    # Early Stopping and ReduceLROnPlateau Logic
    if val_acc > best_val_acc + min_delta:
        best_val_acc = val_acc
        early_stopping_counter = 0
        reduce_lr_counter = 0
        torch.save(model.state_dict(), "best_val_model.pth")
    else:
        early_stopping_counter += 1
        reduce_lr_counter += 1

    # Reduce learning rate on plateau
        if reduce_lr_counter >= patience_lr:
            new_lr = max(current_lr * factor, min_lr)
            if new_lr < current_lr:  # Only update if reducing
                print(f"Reducing learning rate from {current_lr} to {new_lr}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                current_lr = new_lr
            reduce_lr_counter = 0

        # Early stopping
        if early_stopping_counter >= patience_es:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    
    # Test Loop
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100. * correct / total
    print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_acc:.2f}%")

    # Save the best model based on test accuracy
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), "best_test_model.pth")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%, Best Test Accuracy: {best_test_acc:.2f}%")