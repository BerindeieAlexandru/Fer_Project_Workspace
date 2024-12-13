import dlib
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from mtcnn import MTCNN
import matplotlib.pyplot as plt

detector = MTCNN()
def detect_face_with_mtcnn(image):
    results = detector.detect_faces(image)
    if len(results) > 0:
        box = results[0]['box']
        x, y, w, h = box
        face = dlib.rectangle(x, y, x+w, y+h)
        return face
    return None
# Load the facial landmark predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
# detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def align_face(image, landmarks, crop=True):
    """
    Align face based on 5-point landmarks (compatible with MTCNN).
    """
    desired_left_eye = (0.35, 0.35)
    desired_right_eye = (0.65, 0.35)
    desired_face_width = 48
    desired_face_height = 48

    # Use MTCNN's 5 landmarks (left eye, right eye, nose, mouth_left, mouth_right)
    left_eye_center = landmarks[0]
    right_eye_center = landmarks[1]

    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = (desired_right_eye[0] - desired_left_eye[0]) * desired_face_width
    scale = desired_dist / dist

    eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2.0,
               (left_eye_center[1] + right_eye_center[1]) / 2.0)
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    tX = desired_face_width * 0.5
    tY = desired_face_height * desired_left_eye[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    aligned_face = cv2.warpAffine(image, M, (desired_face_width, desired_face_height), flags=cv2.INTER_CUBIC)

    if crop:
        # Crop to avoid black edges
        x_start = max(0, int(eyes_center[0] - desired_face_width * 0.5))
        y_start = max(0, int(eyes_center[1] - desired_face_height * 0.5))
        x_end = x_start + desired_face_width
        y_end = y_start + desired_face_height
        aligned_face = aligned_face[y_start:y_end, x_start:x_end]

    return aligned_face

def test_alignment_on_image(image_path):
    """
    Test face alignment with cropping on a single image.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or invalid format.")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image_rgb)
    
    if len(results) > 0:
        # Detect landmarks and align
        box = results[0]['box']
        keypoints = results[0]['keypoints']
        landmarks = np.array([
            keypoints['left_eye'],
            keypoints['right_eye'],
            keypoints['nose'],
            keypoints['mouth_left'],
            keypoints['mouth_right']
        ])
        aligned_image = align_face(image, landmarks)
    else:
        print("No face detected in the image.")
        return

    aligned_image_rgb = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
    
    # Display the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(aligned_image_rgb)
    plt.title("Aligned and Cropped Image")
    plt.axis("off")

    plt.show()

# Try it on a single image
test_alignment_on_image(r"D:\Alex\Desktop\datasets_processing\fer2013_aligned\fer2013_original_renamed\train\angry\train_5_angry.jpg")