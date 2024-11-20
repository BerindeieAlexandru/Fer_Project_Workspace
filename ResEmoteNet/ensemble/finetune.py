import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import torch.optim as optim
from approach.ResEmoteNet import ResEmoteNet
from get_dataset import Four4All
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR

device = "cuda"
print(f"Using {device} device")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load datasets
train_dataset = Four4All(csv_file=r'D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\ensemble\train_labels.csv',
                         img_dir=r'D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\ensemble\fer2013_o_r\train', transform=transform)

val_dataset = Four4All(csv_file=r'D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\ensemble\val_labels.csv',
                       img_dir=r'D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\ensemble\fer2013_o_r\val', transform=transform)

test_dataset = Four4All(csv_file=r'D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\ensemble\test_labels.csv',
                        img_dir=r'D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\ensemble\fer2013_o_r\test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load model
model = ResEmoteNet().to(device)
model.load_state_dict(torch.load('../snapshots/best_fine_tuned_model.pth'))

# Hyperparameters
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
# scheduler = CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-3, step_size_up=len(fine_tune_loader), mode='exp_range')
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

best_test_accuracy = 0.0

# Fine-tuning settings
num_epochs = 50

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
test_losses = []
test_accuracies = []

# Fine-tuning loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
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
    if test_acc > best_test_accuracy:
        best_test_accuracy = test_acc
        torch.save(model.state_dict(), 'best_fine_tuned_model.pth')

    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_acc}, Test Loss: {test_loss}, Test Accuracy: {test_acc}")

# Save results to a CSV file
df = pd.DataFrame({
    'Epoch': range(1, num_epochs + 1),
    'Train Loss': train_losses,
    'Train Accuracy': train_accuracies,
    'Validation Loss': val_losses,
    'Validation Accuracy': val_accuracies,
    'Test Loss': test_losses,
    'Test Accuracy': test_accuracies
})
df.to_csv('fine_tuning_results.csv', index=False)

