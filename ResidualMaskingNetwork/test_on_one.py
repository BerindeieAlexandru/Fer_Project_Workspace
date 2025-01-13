import json
import os
import cv2
import torch
from torchvision.transforms import transforms
from models import resmasking_dropout1
import numpy as np
import urllib.request

# Download haarcascade xml if not present locally
haar_url = "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_alt.xml"
haar_path = "./haarcascade_frontalface_alt.xml"
if not os.path.exists(haar_path):
    print(f"Downloading Haar Cascade file to {haar_path}...")
    urllib.request.urlretrieve(haar_url, haar_path)

face_cascade = cv2.CascadeClassifier(haar_path)

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

def ensure_color(image):
    if len(image.shape) == 2:
        return np.dstack([image] * 3)
    elif image.shape[2] == 1:
        return np.dstack([image] * 3)
    return image

def ensure_gray(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        pass
    return image


FER_2013_EMO_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

def main(image_path):
    # load configs and set random seed
    configs = json.load(open("configs/fer2013_config.json"))
    image_size = (configs["image_size"], configs["image_size"])

    # model = densenet121(in_channels=3, num_classes=7)
    model = resmasking_dropout1(in_channels=3, num_classes=7)
    model.cuda()

    state = torch.load("checkpoint/resmasking_dropout1_rot30_2019Nov17_14.33")
    model.load_state_dict(state["net"])
    model.eval()

    image = cv2.imread(image_path)

    faces = face_cascade.detectMultiScale(image, 1.15, 5)
    gray = ensure_gray(image)
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (179, 255, 179), 2)

        face = gray[y : y + h, x : x + w]
        face = ensure_color(face)

        face = cv2.resize(face, image_size)
        face = transform(face).cuda()
        face = torch.unsqueeze(face, dim=0)

        output = torch.squeeze(model(face), 0)
        proba = torch.softmax(output, 0)

        emo_proba, emo_idx = torch.max(proba, dim=0)
        emo_idx = emo_idx.item()
        emo_proba = emo_proba.item()

        emo_label = FER_2013_EMO_DICT[emo_idx]

        # Print to console
        print(f"Predicted Emotion: {emo_label}, Probability: {emo_proba * 100:.2f}%")
        
        # Draw label and border on the image
        label_size, base_line = cv2.getTextSize(
            "{}: 000".format(emo_label), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        cv2.rectangle(
            image,
            (x + w, y + 1 - label_size[1]),
            (x + w + label_size[0], y + 1 + base_line),
            (223, 128, 255),
            cv2.FILLED,
        )
        cv2.putText(
            image,
            "{}: {}".format(emo_label, int(emo_proba * 100)),
            (x + w, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
    
    # Resize the image for better visibility
    enlarged_image = cv2.resize(image, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)
    
    # Display the resulting image
    cv2.imshow("Emotion Detection", enlarged_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys

    argv = sys.argv[1]
    assert isinstance(argv, str) and os.path.exists(argv)
    main(argv)
