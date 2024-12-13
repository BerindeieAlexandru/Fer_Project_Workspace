from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import paths
import argparse
import imutils
import dlib
import cv2
import os


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-i", "--input", required=True,
	            help="path to input image")
ap.add_argument("-o", "--output", required=True,
                help="path to output directory where aligned images will be saved")
args = vars(ap.parse_args())

if args == None:
    raise Exception("could not load image !")


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=48, desiredFaceHeight=48)

if not os.path.exists(args["output"]):
    os.makedirs(args["output"])

print("Image pre-processing is starting. Aligning image according to facial landmarks.")
# loop over the input images
for inputPath in paths.list_images(args["input"]):
    # load the image, convert it to grayscale, and describe it
    image = cv2.imread(inputPath)
    print(inputPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale image
    rects = detector(gray, 2)

    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks

        shape = predictor(gray, rect)
        faceAligned = fa.align(image, gray, rect)

        filename = os.path.basename(inputPath)
        outputPath = os.path.join(args["output"], filename)

        cv2.imwrite(outputPath, faceAligned)

        print(f"Processed and saved: {outputPath}")

print("Image face alignment is completed.")