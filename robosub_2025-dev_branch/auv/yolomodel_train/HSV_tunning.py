import cv2
import numpy as np

video_path = "C:/Users/huytr/Documents/GitHub/robosub_2025-dev_branch_test/robosub_2025-dev_branch/auv/yolomodel_train/CV_training_data/CV_1.mp4"  # Change if needed

def nothing(x): pass

cv2.namedWindow("HSV Calibration", cv2.WINDOW_NORMAL)

# Lower red range trackbars
cv2.createTrackbar("LH1", "HSV Calibration", 0, 179, nothing)
cv2.createTrackbar("LS1", "HSV Calibration", 100, 255, nothing)
cv2.createTrackbar("LV1", "HSV Calibration", 100, 255, nothing)
cv2.createTrackbar("UH1", "HSV Calibration", 10, 179, nothing)
cv2.createTrackbar("US1", "HSV Calibration", 255, 255, nothing)
cv2.createTrackbar("UV1", "HSV Calibration", 255, 255, nothing)

# Upper red range trackbars
cv2.createTrackbar("LH2", "HSV Calibration", 170, 179, nothing)
cv2.createTrackbar("LS2", "HSV Calibration", 100, 255, nothing)
cv2.createTrackbar("LV2", "HSV Calibration", 100, 255, nothing)
cv2.createTrackbar("UH2", "HSV Calibration", 179, 179, nothing)
cv2.createTrackbar("US2", "HSV Calibration", 255, 255, nothing)
cv2.createTrackbar("UV2", "HSV Calibration", 255, 255, nothing)

cap = cv2.VideoCapture(video_path)
paused = False
frame_idx = 0

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = frame.copy()
        frame_idx += 1
    else:
        frame = current_frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Get lower red range
    l_h1 = cv2.getTrackbarPos("LH1", "HSV Calibration")
    l_s1 = cv2.getTrackbarPos("LS1", "HSV Calibration")
    l_v1 = cv2.getTrackbarPos("LV1", "HSV Calibration")
    u_h1 = cv2.getTrackbarPos("UH1", "HSV Calibration")
    u_s1 = cv2.getTrackbarPos("US1", "HSV Calibration")
    u_v1 = cv2.getTrackbarPos("UV1", "HSV Calibration")
    lower1 = np.array([l_h1, l_s1, l_v1])
    upper1 = np.array([u_h1, u_s1, u_v1])
    # Get upper red range
    l_h2 = cv2.getTrackbarPos("LH2", "HSV Calibration")
    l_s2 = cv2.getTrackbarPos("LS2", "HSV Calibration")
    l_v2 = cv2.getTrackbarPos("LV2", "HSV Calibration")
    u_h2 = cv2.getTrackbarPos("UH2", "HSV Calibration")
    u_s2 = cv2.getTrackbarPos("US2", "HSV Calibration")
    u_v2 = cv2.getTrackbarPos("UV2", "HSV Calibration")
    lower2 = np.array([l_h2, l_s2, l_v2])
    upper2 = np.array([u_h2, u_s2, u_v2])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key == ord(' '):  # Space to pause/play
        paused = not paused
    elif key == ord('d'):  # Next frame
        paused = True
        ret, frame = cap.read()
        if ret:
            current_frame = frame.copy()
            frame_idx += 1
    elif key == ord('a'):  # Previous frame (reloads; slow)
        paused = True
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx-2))
        ret, frame = cap.read()
        if ret:
            current_frame = frame.copy()
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

cap.release()
cv2.destroyAllWindows()
