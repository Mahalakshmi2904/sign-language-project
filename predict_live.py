import cv2
import mediapipe as mp
import pickle
import numpy as np
import time

# Load trained model
with open("isl_knn_model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

sentence = ""
cursor = 0

current_char = ""
stable_char = ""
char_start_time = time.time()
STABLE_TIME = 1.5  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    predicted_char = ""

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        #NORMALIZATION
        wrist = lm[0]
        data = []

        for p in lm:
            data.append(p.x - wrist.x)
            data.append(p.y - wrist.y)

        data = np.array(data)

        if np.max(np.abs(data)) != 0:
            data = data / np.max(np.abs(data))

        data = data.reshape(1, -1)

        predicted_char = model.predict(data)[0]

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Stability logic
        if predicted_char != current_char:
            current_char = predicted_char
            char_start_time = time.time()
        else:
            if time.time() - char_start_time >= STABLE_TIME:
                #  Allow repeated letters after hand reset
                sentence = sentence[:cursor] + predicted_char + sentence[cursor:]
                cursor += 1
                stable_char = predicted_char
                current_char = ""  # reset to allow repetition

    else:
        #  Reset when hand not visible (important for repeated letters)
        stable_char = ""
        current_char = ""

    #  MULTI-LINE DISPLAY
    display_sentence = sentence[:cursor] + "|" + sentence[cursor:]

    max_chars_per_line = 20
    lines = [display_sentence[i:i+max_chars_per_line]
             for i in range(0, len(display_sentence), max_chars_per_line)]

    # Show detected character
    cv2.putText(frame, f"Detected: {predicted_char}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show wrapped sentence
    y_position = 90
    for line in lines:
        cv2.putText(frame, line, (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 0), 2)
        y_position += 40

    cv2.imshow("ISL Sentence Formation", frame)

    key = cv2.waitKey(1) & 0xFF

    # Cursor left
    if key == ord('a'):
        if cursor > 0:
            cursor -= 1

    # Cursor right
    elif key == ord('d'):
        if cursor < len(sentence):
            cursor += 1

    # Backspace
    elif key == 8:
        if cursor > 0:
            sentence = sentence[:cursor-1] + sentence[cursor:]
            cursor -= 1

    # Insert SPACE
    elif key == 32:
        sentence = sentence[:cursor] + " " + sentence[cursor:]
        cursor += 1

    # Exit
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
