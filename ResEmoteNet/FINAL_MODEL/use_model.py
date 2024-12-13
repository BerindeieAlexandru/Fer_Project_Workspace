import torch
import cv2
import numpy as np
from torchvision import transforms
from mediapipe import solutions
from arhitecture import FourforAll
from PIL import Image

# Initialize Mediapipe Face Detection
mp_face_detection = solutions.face_detection
mp_drawing = solutions.drawing_utils

# Transform pipeline for the model
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Label mapping for emotions
emotion_labels = ['Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear', 'Neutral']

def load_model(device):
    model = FourforAll()
    checkpoint = torch.load('fer_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def preprocess_face(image, bbox):
    x_min, y_min, width, height = bbox
    x_min, y_min, width, height = int(x_min), int(y_min), int(width), int(height)

    # Crop the face from the image
    face = image[y_min:y_min+height, x_min:x_min+width]

    if face.size == 0:  # Avoid empty faces
        return None

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    face = Image.fromarray(face)  # Convert to PIL Image
    face_tensor = transform(face).unsqueeze(0)  # Apply transforms and add batch dimension
    return face_tensor

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    cap = cv2.VideoCapture(0)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break

            # Convert the frame to RGB for Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    bbox = (bboxC.xmin * iw, bboxC.ymin * ih, bboxC.width * iw, bboxC.height * ih)

                    # Preprocess face
                    face_tensor = preprocess_face(frame, bbox)
                    if face_tensor is None:
                        continue

                    # Predict the emotion
                    with torch.no_grad():
                        face_tensor = face_tensor.to(device)
                        outputs = model(face_tensor)
                        _, predicted = torch.max(outputs, 1)
                        emotion = emotion_labels[predicted.item()]

                    # Draw the bounding box and emotion label
                    x_min, y_min, width, height = map(int, bbox)
                    cv2.rectangle(frame, (x_min, y_min), (x_min + width, y_min + height), (255, 0, 0), 2)
                    cv2.putText(frame, emotion, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            cv2.imshow('Real-time Emotion Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()