import cv2
import mediapipe as mp
import csv
import os
import numpy as np
from collections import defaultdict

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

file_name = "isl_alphabets_webcam.csv"

counts = defaultdict(int)

with open(file_name, 'a', newline='') as f:
    writer = csv.writer(f)

    print("Press A-Z to save | ESC to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            lm = hand.landmark

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            wrist = lm[0]
            data = []

            for p in lm:
                data.append(p.x - wrist.x)
                data.append(p.y - wrist.y)

            data = np.array(data)

            if np.max(np.abs(data)) != 0:
                data = data / np.max(np.abs(data))

            cv2.putText(
                frame,
                "Press A-Z to save | ESC to quit",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow("Collect ISL Alphabets", frame)

        key = cv2.waitKey(1) & 0xFF

        if key >= ord('a') and key <= ord('z') and result.multi_hand_landmarks:
            label = chr(key).upper()
            writer.writerow(data.tolist() + [label])

            counts[label] += 1
            print(f"{label} count: {counts[label]}")

        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()

print("\nFinal Counts:")
for k, v in counts.items():
    print(f"{k}: {v}")
