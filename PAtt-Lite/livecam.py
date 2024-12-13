import torch
import cv2
import numpy as np
from torchvision import transforms
from approach_mnv3 import *  # Ensure you import your custom model correctly

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model_path = r"best_model.pth"  # Path to your trained model
num_classes = 7  # Number of emotion classes
model = CustomModel(num_classes=num_classes, dropout_rate=0.2)
model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device))
model = model.to(device)
model.eval()

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Define preprocessing pipeline
preprocess = transforms.Compose([
    transforms.ToPILImage(),       # Convert OpenCV image (NumPy array) to PIL Image
    transforms.Resize((224, 224)), # Resize to model's input size
    transforms.ToTensor(),         # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Initialize Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Check if the camera is opened
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

print("Press 'q' to quit.")

# Real-time inference
while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        print("Error: Unable to read from the webcam.")
        break

    # Convert BGR (OpenCV default) to grayscale (Haar Cascade works with grayscale images)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no faces are detected, continue the loop
    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Crop the first detected face (you can modify this to handle multiple faces if needed)
        (x, y, w, h) = faces[0]  # Get the coordinates of the first face
        face = frame[y:y+h, x:x+w]  # Crop the face from the frame

        # Convert BGR (OpenCV) to RGB
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Preprocess the cropped face
        try:
            input_tensor = preprocess(rgb_face).unsqueeze(0).to(device)  # Add batch dimension
        except Exception as e:
            print(f"Preprocessing error: {e}")
            continue

        # Make predictions
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion = emotion_labels[predicted.item()]

        # Overlay emotion on the frame
        cv2.putText(frame, f"Emotion: {emotion}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Emotion Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
