from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")

# supports raspberry pi camera since it's cheap and also requires less power it is preferred in real world usage
# ap.add_argument("-r", "--picamera", type=int, default=-1,
#                 help="whether or not the Raspberry Pi camera should be used")

args = vars(ap.parse_args())

print("loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
print("camera sensor warming up...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=args["picamera"] > 0).start() # for raspberry pi
time.sleep(2.0)


while True:  # looping over frames in the video stream
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # press 'q' to quit and break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
