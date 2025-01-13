import json
import numpy as np
import torch
import cv2
import mediapipe as mp
import torch.nn.functional as F
import models

# Emotion labels for FER2013 dataset
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Model configuration
model_dict = [("resmasking_dropout1", "resmasking_dropout1_rot30_2019Nov17_14.33")]


def preprocess_face(face, config):
    # Resize and normalize the face image to fit the model's input requirements
    face = cv2.resize(face, (config["image_size"], config["image_size"]))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face / 255.0
    face = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Shape: [1, 3, H, W]
    return face


def main():
    # Load model configuration
    with open("D:\\Alex\\Desktop\\ResidualMaskingNetwork-master\\configs\\fer2013_config.json") as f:
        configs = json.load(f)

    # Initialize the model
    for model_name, checkpoint_path in model_dict:
        model = getattr(models, model_name)(in_channels=3, num_classes=7)
        state = torch.load(r"D:\Alex\Desktop\ResidualMaskingNetwork-master\checkpoint\\" + checkpoint_path)
        model.load_state_dict(state["net"])
        model.to("cuda")
        model.eval()

    # Initialize MediaPipe face detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)

    # Start video capture
    cap = cv2.VideoCapture(0)  # 0 is the default camera

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                # Get bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(
                    bboxC.height * h)

                # Extract the face region
                face = frame[y:y + h_box, x:x + w_box]

                # Only proceed if face region is within the frame bounds
                if face.size == 0:
                    continue

                # Preprocess the face
                face_tensor = preprocess_face(face, configs).to("cuda")

                # Predict emotion
                with torch.no_grad():
                    result = model(face_tensor)
                    result = F.softmax(result, dim=1)
                    probabilities = result.cpu().numpy()[0]

                    # Get the highest probability emotion
                    predicted_emotion_idx = np.argmax(probabilities)
                    predicted_emotion = emotion_labels[predicted_emotion_idx]

                # Display the predicted emotion and probability on the frame
                label = f"{predicted_emotion} ({probabilities[predicted_emotion_idx]:.2f})"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Emotion Recognition', frame)

        # Press 'q' to exit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()


if __name__ == "__main__":
    main()
