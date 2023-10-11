from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="path to input video file")
args = vars(ap.parse_args())

stream = cv2.VideoCapture(args["video"])
# timer to measure FPS, or more specifically, the throughput rate of our video processing pipeline.
fps = FPS().start()


while True:
    (grabbed, frame) = stream.read()
    # if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break
    frame = imutils.resize(frame, width=450)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])

    cv2.putText(frame, "Slow Method", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # show the frame and update the FPS counter
    cv2.imshow("Frame", frame)
    # delay of 1ms and bitwise operation to consider only last 8-bits allows compatibilty with both 32-bit and 64-bit systems
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("elasped time: {:.2f}".format(fps.elapsed()))
print("approx. FPS: {:.2f}".format(fps.fps()))

stream.release()
cv2.destroyAllWindows()
