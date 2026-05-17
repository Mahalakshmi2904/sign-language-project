import cv2
import numpy as np

cap = cv2.VideoCapture(0)
bg = None

x1, y1, x2, y2 = 50, 50, 400, 400

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    roi = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    key = cv2.waitKey(1) & 0xFF

    # Press b to capture background
    if key == ord('b'):
        bg = gray.copy()
        print("Background captured")

    hand = False

    if bg is not None:

        diff = cv2.absdiff(bg, gray)

        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5,5), np.uint8)

        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            kernel
        )

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower = (0, 30, 60)
        upper = (20, 170, 255)

        skin = cv2.inRange(hsv, lower, upper)

        hand_mask = cv2.bitwise_and(thresh, skin)

        white_pixels = cv2.countNonZero(hand_mask)

        # Improved detection
        hand = white_pixels > 1000

        cv2.imshow(
            "ROI - Hand",
            cv2.resize(hand_mask, (400, 400))
        )

    text = "Hand detected" if hand else "No hand"

    color = (0,255,0) if hand else (0,0,255)

    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

    cv2.putText(
        frame,
        text,
        (x1, y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    )

    cv2.imshow("Camera", frame)

    if key == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()