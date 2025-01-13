import cv2
from rmn import RMN

def mark_face_and_write_text(image_path, text_emo, output_path):
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not open or find the image: {image_path}")
        return

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (199, 255, 47), 3)

        # Write 'happy' above the face with larger text
        font_scale = 3  # Increase the font scale to make the text larger
        font_thickness = 5  # Increase the thickness for better visibility
        text = text_emo
        text_color = (199, 255, 47)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = x + (w - text_size[0]) // 2  # Center the text above the face
        text_y = y - 10 if y - 10 > 20 else y + text_size[1] + 10  # Ensure text is within image bounds

        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    # Save the output image
    cv2.imwrite(output_path, image)

    print(f"Output image saved at: {output_path}")

# # Example usage
# image_path = 'eu.jpg'
# output_path = 'output.jpg'
# mark_face_and_write_text(image_path, output_path)

m = RMN()
m.video_demo()