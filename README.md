# Faces
Visual Perception Toolkit: Real-Time Drowsiness Detection, Facial Landmark Prediction, Blink Detection, and Face Identification with Dlib.

# Drowsiness Detection 

https://github.com/AffaanShaikh/Faces/assets/130907730/8b1cb95d-d6d4-43e8-b374-30b6c1f2d8d2

This script utilizes advanced Computer Vision techniques to monitor a person's level of alertness in real-time. By leveraging facial landmarks and calculating the Eye Aspect Ratio (EAR), it accurately identifies whether the person's eyes are open or closed. This is particularly valuable in scenarios where staying awake and alert is critical, such as during long drives or in environments where attentiveness is paramount.

Methods and Techniques:
1) Eye Aspect Ratio (EAR): EAR is a measure used to determine if a person's eyes are open or closed. It's a ratio of distances between various points on the eye. It is a reliable metric for determining eye openness. By measuring the relative distances between key points on the eyes, the script accurately discerns between open and closed eyes.
2) Multi-threaded Alarm System: The script employs multi-threading to ensure that the alarm sound is played concurrently with the main execution, guaranteeing seamless operation without interruption. Using a separate thread for playing the alarm is important for a few reasons: Non-Blocking Execution, Continuous Monitoring, and better User Experience. The alarm thread is set as a daemon thread. A daemon thread is a type of thread that doesn't prevent the program from exiting.
3) Real-time Monitoring and Visual Feedback: The script continuously monitors the person's eyes, allowing for immediate detection and timely alerts in case of drowsiness. The script employs convex hulls to visualize and define the shape of the eyes. This is used to draw contours around the eyes. This feature is invaluable for fine-tuning and debugging.

The drowsiness detector serves as a vital safeguard against potential accidents caused by fatigue-induced lapses in alertness. By utilizing advanced Computer Vision techniques, this tool continuously monitors a person's eyes in real-time, accurately discerning between open and closed states. This is particularly critical in contexts where sustained attentiveness is paramount, such as during long drives or in environments where alertness is essential for safety. The system's ability to calculate the Eye Aspect Ratio (EAR) provides an objective measure of eye openness, ensuring accurate detection of drowsiness. In instances where prolonged eye closure is detected, the system triggers an alarm, promptly alerting the individual to regain focus. This feature has far-reaching implications, not only in automotive safety but also in industrial settings where worker vigilance is crucial. Overall, the drowsiness detector stands as a powerful tool in averting potentially disastrous situations by proactively addressing the risks associated with drowsy states.

Usage Instructions:

1) Install the required libraries and dependencies.
2) Run the script, providing the path to the facial landmark predictor file using the --shape-predictor argument.
3) Optionally, specify an alarm sound file using the --alarm argument.
4) Adjust the threshold and frame counter variables (EYE_AR_THRESH and EYE_AR_CONSEC_FRAMES) to suit your specific needs.
