from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()  # dlib's face detector (HOG-based)
predictor = dlib.shape_predictor(
    args["shape_predictor"])  # facial landmark predictor

# load the input image, resize it, and convert it to grayscale and then detect faces in the grayscale image
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region
    shape = predictor(gray, rect)
    # convert the facial landmark (x, y)-coordinates to a NumPy array
    shape = face_utils.shape_to_np(shape)

    # rect to bb in the format:(x, y, w, h) then draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # display the face number
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

# dsiplay output image with the face detections and facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)
