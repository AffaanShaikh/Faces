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

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)


for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # loop over the face parts individually
    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        # clone the original image so we can draw on it, then
        # display the name of the face part on the image
        clone = image.copy()
        cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        # loop over the subset of facial landmarks, drawing the specific face part
        for (x, y) in shape[i:j]:
            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

        # extract the ROI(region of interest) of the face region as a separate image
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        roi = image[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

        # display specific face part
        cv2.imshow("ROI", roi)
        cv2.imshow("Image", clone)
        cv2.waitKey(0)

    # visualize all facial landmarks with a transparent overlay
    output = face_utils.visualize_facial_landmarks(image, shape)
    cv2.imshow("Image", output)
    cv2.waitKey(0)
