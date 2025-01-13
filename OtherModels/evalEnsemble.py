import torch
import os
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights, efficientnet_v2_m, EfficientNet_V2_M_Weights, resnext50_32x4d
import torch.nn as nn
from PIL import Image
import timm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from ResEmoteNet import model_arhitecture as arch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

def load_mobilenetv3(num_classes, device, checkpointname='MobileNetV3/mobilenetv3_best.pth'):
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load(checkpointname, weights_only=True)['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def load_mobilenetv4(num_classes, device, checkpointname='MobileNetV4/mobilenetv4_best.pth'):
    model = timm.create_model('mobilenetv4_hybrid_large.e600_r384_in1k', pretrained=True, num_classes=num_classes)
    checkpoint = torch.load(checkpointname, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def load_efficientnetv2(num_classes, device, checkpointname='EfficientNet_V2M/efficientnet_v2m_best.pth'):
    model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(checkpointname, weights_only=True)['model_state_dict']) 
    model = model.to(device)
    model.eval()
    return model

def load_resnext50_32x4d(num_classes, device, checkpointname='ResNeXt50_32x4d/resnext50_32x4d_best.pth'):
    model = resnext50_32x4d(weights="ResNeXt50_32X4D_Weights.IMAGENET1K_V2")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    checkpoint = torch.load(checkpointname, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def load_fbnetv3b(num_classes, device, checkpointname='FBNetV3b/fbnetv3b_best.pth'):
    model = timm.create_model('fbnetv3_b.ra2_in1k', pretrained=True, num_classes=num_classes)
    checkpoint = torch.load(checkpointname, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def load_resemotenet(num_classes, device, checkpointname='ResEmoteNet/ResEmoteNet_best.pth'):
    model = arch.ResEmoteNet()
    checkpoint = torch.load(checkpointname, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def plot_roc_curve(y_true_bin, y_pred_prob, class_mapping):
    plt.figure(figsize=(10, 8))
    for idx, emotion in enumerate(class_mapping.keys()):
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_pred_prob[:, idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{emotion} (ROC-AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for each class')
    plt.legend(loc='lower right')
    plt.savefig("roc_curve.png")
    plt.show()

def plot_pr_curve(y_true_bin, y_pred_prob, class_mapping):
    plt.figure(figsize=(10, 8))
    for idx, emotion in enumerate(class_mapping.keys()):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, idx], y_pred_prob[:, idx])
        pr_auc = average_precision_score(y_true_bin[:, idx], y_pred_prob[:, idx])
        plt.plot(recall, precision, lw=2, label=f'{emotion} (PR-AUC = {pr_auc:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for each class')
    plt.legend(loc='lower left')
    plt.savefig("pr_curve.png")
    plt.show()

def evaluate_ensemble(folder_path, class_mapping, device):
    device = device

    num_classes = len(class_mapping)
    models = [
        # load_fbnetv3b(num_classes, device),
        # load_mobilenetv3(num_classes, device),
        # load_resnext50_32x4d(num_classes, device),
        # load_efficientnetv2(num_classes, device),
        load_resemotenet(num_classes, device, r"ResEmoteNet\ResEmoteNet_best.pth"),
        # load_mobilenetv4(num_classes, device),
    ]

    use_tim = False

    # Define the transformation: if we use timm then use special config, otherwise use imagenet one and for own model use custom one
    if use_tim :
        data_config = timm.data.resolve_data_config(models[0].pretrained_cfg)
        transforms_imagenet_eval = timm.data.create_transform(**data_config, is_training=False)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms_imagenet_eval,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        ])
    custom_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    # Results lists
    true_labels = []
    predicted_labels = []
    # Logits for each class for roc and pr curves
    all_logits = []

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        # Extract the true label from the file name
        try:
            emotion_name = image_name.split('_')[-1].split('.')[0]
            if emotion_name not in class_mapping:
                print(f"Skipping file '{image_name}' with unknown emotion '{emotion_name}'")
                continue
            true_label = class_mapping[emotion_name]
        except Exception as e:
            print(f"Error extracting emotion from '{image_name}': {e}")
            continue

        try:
            image = Image.open(image_path).convert('RGB')
            logits_sum = None

            # If model is ResEmoteNet, use custom transformation otherwise use the standard ones
            for model in models:
                if isinstance(model, arch.ResEmoteNet):
                    image_tensor = custom_transform(image).unsqueeze(0).to(device)
                else:
                    image_tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(image_tensor)

                    if logits_sum is None:
                        logits_sum = outputs
                    else:
                        logits_sum += outputs

            averaged_logits = logits_sum / len(models)
            all_logits.append(averaged_logits.cpu().numpy().flatten())
            _, predicted = torch.max(averaged_logits, 1)
            true_labels.append(true_label)
            predicted_labels.append(predicted.item())
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            continue

    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, target_names=class_mapping.keys(), zero_division=0, output_dict=True)

    print(f"Overall Accuracy: {accuracy}")
    print("Classification Report:\n")
    print(classification_report(true_labels, predicted_labels, target_names=class_mapping.keys(), zero_division=0))

    # Save classification report to CSV
    report_df = pd.DataFrame(report).transpose()
    report_df['accuracy'] = accuracy
    report_df.to_csv("classification_report.csv", index=True)

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_mapping.keys(), yticklabels=class_mapping.keys())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("confusion_matrix.png")

    # Compute ROC-AUC and PR-AUC for each class and plot them
    y_true_bin = label_binarize(true_labels, classes=list(range(num_classes)))
    y_pred_prob = np.array(all_logits)
    roc_auc = {}
    pr_auc = {}

    for idx, emotion in enumerate(class_mapping.keys()):
        roc_auc[emotion] = roc_auc_score(y_true_bin[:, idx], y_pred_prob[:, idx])
        pr_auc[emotion] = average_precision_score(y_true_bin[:, idx], y_pred_prob[:, idx])

    print("ROC-AUC per class:")
    for emotion, score in roc_auc.items():
        print(f"  {emotion}: {score:.4f}")

    print("\nPR-AUC per class:")
    for emotion, score in pr_auc.items():
        print(f"  {emotion}: {score:.4f}")

    plot_roc_curve(y_true_bin, y_pred_prob, class_mapping)
    plot_pr_curve(y_true_bin, y_pred_prob, class_mapping)

class EmotionDataset(Dataset):
    def __init__(self, folder_path, transform, class_mapping):
        self.image_paths = []
        self.labels = []
        self.class_mapping = class_mapping
        self.transform = transform

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            try:
                emotion_name = image_name.split('_')[-1].split('.')[0]
                if emotion_name not in class_mapping:
                    continue
                true_label = class_mapping[emotion_name]
                self.image_paths.append(image_path)
                self.labels.append(true_label)
            except Exception as e:
                print(f"Error processing file '{image_name}': {e}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image at {image_path}: {e}")
            return None, label

# For this we cannot use ResEmoteNet because it has a different input size than rest
def evaluate_ensemble_with_batches(folder_path, class_mapping, device, batch_size=16):
    device = device

    num_classes = len(class_mapping)
    models = [
        # load_fbnetv3b(num_classes, device),
        # load_mobilenetv3(num_classes, device),
        # load_resnext50_32x4d(num_classes, device),
        # load_efficientnetv2(num_classes, device),
        # load_mobilenetv4(num_classes, device),
        # load_resemotenet(num_classes, device),
    ]

    use_tim = True

    # Same as above for transforms
    if use_tim:
        data_config = timm.data.resolve_data_config(models[0].pretrained_cfg)
        transforms_imagenet_eval = timm.data.create_transform(**data_config, is_training=False)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms_imagenet_eval,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    custom_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load data in batches for faster processing
    dataset = EmotionDataset(folder_path, transform, class_mapping)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    true_labels = []
    predicted_labels = []
    all_logits = []

    for batch_images, batch_labels in tqdm(dataloader, desc="Evaluating images", unit="batch"):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        batch_logits_sum = None

        # Aggregate logits from all models
        for model in models:
            model.eval()
            with torch.no_grad():
                if isinstance(model, arch.ResEmoteNet):
                    model_logits = model(batch_images)
                else:
                    model_logits = model(batch_images)
                
                if batch_logits_sum is None:
                    batch_logits_sum = model_logits
                else:
                    batch_logits_sum += model_logits

        # Average logits
        averaged_logits = batch_logits_sum / len(models)
        all_logits.extend(averaged_logits.cpu().numpy())
        _, batch_predictions = torch.max(averaged_logits, 1)

        true_labels.extend(batch_labels.cpu().numpy())
        predicted_labels.extend(batch_predictions.cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, target_names=class_mapping.keys(), zero_division=0, output_dict=True)

    print(f"Overall Accuracy: {accuracy}")
    print("Classification Report:\n")
    print(classification_report(true_labels, predicted_labels, target_names=class_mapping.keys(), zero_division=0))

    report_df = pd.DataFrame(report).transpose()
    report_df['accuracy'] = accuracy
    report_df.to_csv("classification_report.csv", index=True)

    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_mapping.keys(), yticklabels=class_mapping.keys())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("confusion_matrix.png")

    # Compute ROC-AUC and PR-AUC for each class
    y_true_bin = label_binarize(true_labels, classes=list(range(num_classes)))
    y_pred_prob = np.array(all_logits)
    roc_auc = {}
    pr_auc = {}

    for idx, emotion in enumerate(class_mapping.keys()):
        roc_auc[emotion] = roc_auc_score(y_true_bin[:, idx], y_pred_prob[:, idx])
        pr_auc[emotion] = average_precision_score(y_true_bin[:, idx], y_pred_prob[:, idx])

    print("ROC-AUC per class:")
    for emotion, score in roc_auc.items():
        print(f"  {emotion}: {score:.4f}")

    print("\nPR-AUC per class:")
    for emotion, score in pr_auc.items():
        print(f"  {emotion}: {score:.4f}")

    plot_roc_curve(y_true_bin, y_pred_prob, class_mapping)
    plot_pr_curve(y_true_bin, y_pred_prob, class_mapping)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_mapping = {
        'happy': 0,
        'surprise': 1,
        'sad': 2,
        'angry': 3,
        'disgust': 4,
        'fear': 5,
        'neutral': 6
    }

    folder_path = r"D:\Alex\Desktop\Eperiments\datasets_processing\fer2013original_workspace\fer2013_original_restructured\test"

    # Ensemble for all models
    # evaluate_ensemble(folder_path, class_mapping, device)

    # Emsemble for all models excluding ResEmoteNet
    evaluate_ensemble_with_batches(folder_path, class_mapping, device)

if __name__ == "__main__":
    main()
