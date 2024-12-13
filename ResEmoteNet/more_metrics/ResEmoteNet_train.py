import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
import torch.optim as optim
from models.ResEmoteNet import ResEmoteNet
from get_dataset import Four4All
from scheduler import CosineAnnealingWithWarmRestartsLR
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.utils import compute_class_weight

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy, all_labels, all_preds

def main():

    device = "cuda"
    print(f"Using {device} device")

    # Transform for train
    train = v2.Compose([
        v2.Resize((64, 64)),
        v2.Grayscale(num_output_channels=3),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(degrees=(-10, 10)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Transform for test
    eval = v2.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load the dataset
    train_dataset = Four4All(csv_file='trainb_labels.csv', img_dir='fer2013_o_f_bal/train', transform=train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    train_image, train_label = next(iter(train_loader))


    val_dataset = Four4All(csv_file='valb_labels.csv', img_dir='fer2013_o_f_bal/val', transform=eval)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    val_image, val_label = next(iter(val_loader))


    test_dataset = Four4All(csv_file='testb_labels.csv', img_dir='fer2013_o_f_bal/test', transform=eval)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_image, test_label = next(iter(test_loader))


    print(f"Train batch: Image shape {train_image.shape}, Label shape {train_label.shape}")
    print(f"Validation batch: Image shape {val_image.shape}, Label shape {val_label.shape}")
    print(f"Test batch: Image shape {test_image.shape}, Label shape {test_label.shape}")

    # Load the labels for the entire training dataset
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())

    train_labels = np.array(train_labels)
    # Compute the class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Computed Class Weights: {class_weights}")

    # Load the model
    model = ResEmoteNet().to(device)

    # Hyperparameters
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingWithWarmRestartsLR(optimizer=optimizer,warmup_steps=128,cycle_steps=1024,min_lr=0.0,max_lr=1e-3)

    patience = 15
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    epoch_counter = 0

    num_epochs = 100

    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []
    val_precision, val_recall, val_f1 = [], [], []
    test_precision, test_recall, test_f1 = [], [], []

    # Start training
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

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

        # Evaluate on validation set
        val_loss, val_acc, val_labels, val_preds = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='macro', zero_division=0)
        val_precision.append(precision)
        val_recall.append(recall)
        val_f1.append(f1)

        # Evaluate on test set
        test_loss, test_acc, test_labels, test_preds = evaluate(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='macro', zero_division=0)
        test_precision.append(precision)
        test_recall.append(recall)
        test_f1.append(f1)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}\n")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n")

        print("\nTest Set Classification Report:")
        print(classification_report(test_labels, test_preds, zero_division=0))
        epoch_counter += 1

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0 
            torch.save(model.state_dict(), '../best_model_val.pth')
        else:
            patience_counter += 1
            print(f"No improvement in validation accuracy for {patience_counter} epochs.")
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), '../best_model_test.pth')
        if patience_counter > patience:
            print("Stopping early due to lack of improvement in validation accuracy.")
            break

    df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train Loss': train_losses,
        'Validation Loss': val_losses,
        'Test Loss': test_losses,
        'Train Accuracy': train_accuracies,
        'Validation Accuracy': val_accuracies,
        'Test Accuracy': test_accuracies,
        'Validation Precision': val_precision,
        'Validation Recall': val_recall,
        'Validation F1-Score': val_f1,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test F1-Score': test_f1
    })
    df.to_csv('run_result.csv', index=False)

if __name__ == '__main__':
    main()