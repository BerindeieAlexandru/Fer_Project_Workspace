import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from approach.ResEmoteNet import ResEmoteNet
from get_dataset import Four4All
from ranger_adabelief import RangerAdaBelief
from torch.optim.lr_scheduler import CyclicLR
import numpy as np


device = "cuda"
print(f"Using {device} device")

# Transform the dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
#     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# Load the dataset

train_dataset = Four4All(csv_file=r'D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\ensemble\train_labels.csv',
                         img_dir=r'D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\ensemble\fer2013_o_r\train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
train_image, train_label = next(iter(train_loader))


val_dataset = Four4All(csv_file=r'D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\ensemble\val_labels.csv',
                       img_dir=r'D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\ensemble\fer2013_o_r\val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
val_image, val_label = next(iter(val_loader))


test_dataset = Four4All(csv_file=r'D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\ensemble\test_labels.csv',
                        img_dir=r'D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\ensemble\fer2013_o_r\test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
test_image, test_label = next(iter(test_loader))

print(f"Train batch: Image shape {train_image.shape}, Label shape {train_label.shape}")
print(f"Validation batch: Image shape {val_image.shape}, Label shape {val_label.shape}")
print(f"Test batch: Image shape {test_image.shape}, Label shape {test_label.shape}")


# # Load the labels for the entire training dataset
# train_labels = []
# for _, labels in train_loader:
#     train_labels.extend(labels.numpy())
#
# train_labels = np.array(train_labels)
# # Compute the class weights
# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
# class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
# print(f"Computed Class Weights: {class_weights}")

# Load the model

model = ResEmoteNet().to(device)

# Print the number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')


# Hyperparameters
# criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
# optimizer = torch.optim.RAdam(model.parameters(), lr=0.001, weight_decay=1e-4)
# optimizer = RangerAdaBelief(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-3, step_size_up=len(train_loader), mode='triangular')

patience = 15
best_val_acc = 0
best_test_acc = 0
patience_counter = 0
epoch_counter = 0

num_epochs = 80
save_epochs = [30, 40, 50, 60, 80]
snapshot_dir = "../snapshots/"
os.makedirs(snapshot_dir, exist_ok=True)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
test_losses = []
test_accuracies = []

# Start training
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        scheduler.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)


    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss = test_running_loss / len(test_loader)
    test_acc = test_correct / test_total
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), '../snapshots/best_test_model.pth')

    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(val_loader)
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_acc}, Test Loss: {test_loss}, Test Accuracy: {test_acc}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
    epoch_counter += 1

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), '../snapshots/best_val_model.pth')
    else:
        patience_counter += 1
        print(f"No improvement in validation accuracy for {patience_counter} epochs.")

        # Save models at predefined epochs
        if (epoch + 1) in save_epochs:
            torch.save(model.state_dict(), os.path.join(snapshot_dir, f'model_epoch_{epoch + 1}.pth'))
            print(f"Model saved at epoch {epoch + 1}.")

    if patience_counter > patience:
        print("Stopping early due to lack of improvement in validation accuracy.")
        break

df = pd.DataFrame({
    'Epoch': range(1, epoch_counter+1),
    'Train Loss': train_losses,
    'Test Loss': test_losses,
    'Validation Loss': val_losses,
    'Train Accuracy': train_accuracies,
    'Test Accuracy': test_accuracies,
    'Validation Accuracy': val_accuracies
})
df.to_csv('run_result.csv', index=False)