import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import datetime
import os
from approach_mnv2 import *

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

# Fine-Tuning Parameters
FT_LR = 1e-4
FT_DROPOUT = 0.3
FT_EPOCH = 50
FT_LR_DECAY_STEP = 1000
FT_LR_DECAY_RATE = 0.95
FT_ES_PATIENCE = 10
ES_LR_MIN_DELTA = 1e-4
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Backbone Model
backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
unfreeze = 59

# Load pre-trained model
model = CustomModel(num_classes=7, dropout_rate=0.3)
model.load_state_dict(torch.load("best_test_model.pth"))


backbone_layers = list(model.backbone.children())
freeze_until = len(backbone_layers) - unfreeze
for i, layer in enumerate(backbone_layers):
    for param in layer.parameters():
        param.requires_grad = i >= freeze_until
    # Freeze BatchNorm layers during fine-tuning
    if isinstance(layer, nn.BatchNorm2d):
        for param in layer.parameters():
            param.requires_grad = False

model = model.to(device)

# Modify dropout for fine-tuning
model.dropout = nn.Dropout(FT_DROPOUT)

# Fine-Tune Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FT_LR)

# Scheduler: Inverse Time Decay
def inverse_time_decay(step):
    return 1 / (1 + (step / FT_LR_DECAY_STEP) * (1 - FT_LR_DECAY_RATE))

scheduler = LambdaLR(optimizer, lr_lambda=inverse_time_decay)

# Fine-Tuning Training Loop
best_test_acc = 0
best_val_acc = 0
early_stopping_counter = 0

for epoch in range(FT_EPOCH):
    # Training Phase
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
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    print(f"Epoch {epoch+1}/{FT_EPOCH}, Train Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%")

    # Validation Phase
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

    # Early Stopping Logic
    if val_acc > best_val_acc + ES_LR_MIN_DELTA:
        best_val_acc = val_acc
        early_stopping_counter = 0
        torch.save(model.state_dict(), "best_finetuned_val_model.pth")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= FT_ES_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Test Evaluation
    model.eval()
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
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), "best_finetuned_test_model.pth")
    scheduler.step()

