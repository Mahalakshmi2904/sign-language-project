import cv2
import mediapipe as mp
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

DATASET_PATH = "dataset - Gesture Speech"
CSV_FILE = "isl_landmarks.csv"

with open(CSV_FILE, 'w', newline='') as f:
    writer = csv.writer(f)

    for label in os.listdir(DATASET_PATH):
        folder = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(folder):
            continue

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0].landmark
                row = []
                for p in lm:
                    row.extend([p.x, p.y])
                row.append(label.upper())
                writer.writerow(row)

print("DONE: CSV created")
