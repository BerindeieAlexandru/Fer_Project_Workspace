import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
import mediapipe as mp 
import timm
from torchvision.models import resnext50_32x4d, efficientnet_v2_m, EfficientNet_V2_M_Weights, efficientnet_b0, EfficientNet_B0_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights

def load_mobilenetv3(num_classes, device):
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load("MobileNetV3/mobilenetv3_best.pth")['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def load_mobilenetv4(num_classes, device):
    model = timm.create_model('mobilenetv4_hybrid_large.e600_r384_in1k', pretrained=True, num_classes=num_classes)
    checkpoint = torch.load('MobileNetV4/mobilenetv4_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def load_resnext50_32x4d(num_classes, device):
    model = resnext50_32x4d(weights="ResNeXt50_32X4D_Weights.IMAGENET1K_V2")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    checkpoint = torch.load('ResNeXt50_32x4d/resnext50_32x4d_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def load_efficientnet_v2m(num_classes, device):
    model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load("EfficientNet_V2M/efficientnet_v2m_best.pth")['model_state_dict']) 
    model = model.to(device)
    model.eval()
    return model

def load_efficientnet_b0(num_classes, device):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load("EfficientNet_B0/efficientnet_b0_best.pth")['model_state_dict']) 
    model = model.to(device)
    model.eval()
    return model 

def load_eva(num_classes, device):
    model = timm.create_model('eva_large_patch14_196.in22k_ft_in22k_in1k', pretrained=True, num_classes=num_classes)
    checkpoint = torch.load('Eva/eva_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def load_fbnetv3b(num_classes, device):
    model = timm.create_model('fbnetv3_b.ra2_in1k', pretrained=True, num_classes=num_classes)
    checkpoint = torch.load('FBNetV3b/fbnetv3b_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def preprocess_frame(face, transform, device):
    face = cv2.resize(face, (224, 224))
    face = transform(face)
    return face.unsqueeze(0).to(device)

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 7
    class_labels = ['Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear', 'Neutral']

    # # MobileNetV3
    # model = load_mobilenetv3(num_classes, device)

    # ResNeXt50_32x4d
    # model = load_resnext50_32x4d(num_classes, device)

    # EfficientNet-V2M
    # model = load_efficientnet_v2m(num_classes, device)

    # EfficientNet-B0
    model = load_efficientnet_b0(num_classes, device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # # MobileNetV4
    # model = load_mobilenetv4(num_classes, device)
    # data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    # transforms_imagenet_eval = timm.data.create_transform(**data_config, is_training=False)

    # # Eva
    # model = load_eva(num_classes, device)
    # data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    # transforms_imagenet_eval = timm.data.create_transform(**data_config, is_training=False)

    # # FBNetV3b
    # model = load_fbnetv3b(num_classes, device)
    # data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    # transforms_imagenet_eval = timm.data.create_transform(**data_config, is_training=False)
    
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms_imagenet_eval
    # ])

    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x1, y1 = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                x2, y2 = int((bboxC.xmin + bboxC.width) * iw), int((bboxC.ymin + bboxC.height) * ih)

                margin_x = int(0.1 * (x2 - x1))
                margin_y = int(0.2 * (y2 - y1))

                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(iw, x2 + margin_x)
                y2 = min(ih, y2)

                face = frame[y1:y2, x1:x2]

                input_tensor = preprocess_frame(face, transform, device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    predicted_class = class_labels[predicted.item()]

                # Draw on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{predicted_class}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Real-Time Emotion Recognition', frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
