import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from approach.ResEmoteNet import ResEmoteNet
from get_dataset import Four4All
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define transform (make sure it's the same as used during training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load the test dataset
test_dataset = Four4All(csv_file=r'test_labels.csv',
                        img_dir=r'fer2013_original_restructured/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
test_image, test_label = next(iter(test_loader))

# List of saved model paths
model_paths = [
    r"D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\fer2013_original_train\snapshots\best_test_model.pth",
    # r"D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\fer2013_balanced_train\snapshots\best_val_model.pth",
    # r"D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\fer2013_balanced_train\snapshots\best_model_30_40.pth",
    # r"D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\fer2013_balanced_train\snapshots\best_model_40_50.pth",
    # r"D:\Alex\Documents\Master\An 2\Dizertatie\ResEmoteNet\fer2013_balanced_train\snapshots\best_model_50_60.pth",
]

# Load the models
models = []

# with weights_only
for path in model_paths:
    model = ResEmoteNet().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    models.append(model)

# # without weights_only
# for path in model_paths:
#     model = ResEmoteNet().to(device)
#     checkpoint = torch.load(path, weights_only=False)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
#     models.append(model)


# Function to get ensemble predictions
def ensemble_predictions(models, loader):
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating Ensemble"):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Collect predictions from each model
            outputs = [model(inputs) for model in models]

            # # Average the predictions (logits) from all models
            # avg_outputs = torch.mean(torch.stack(outputs), dim=0)
            #
            # # Get the final predictions by choosing the class with the highest average score
            # _, predicted = torch.max(avg_outputs, 1)

            # Get predictions from each model
            predictions = [torch.argmax(output, dim=1) for output in outputs]

            # Convert list of predictions to a tensor and compute mode (most common prediction)
            predicted = torch.mode(torch.stack(predictions), dim=0).values

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels)


# Get ensemble predictions on the test set
ensemble_preds, true_labels = ensemble_predictions(models, test_loader)

# Evaluate ensemble performance
accuracy = np.mean(ensemble_preds == true_labels)
print(f"Ensemble Accuracy: {accuracy:.4f}")
