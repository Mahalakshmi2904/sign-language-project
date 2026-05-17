import cv2
import mediapipe as mp
import joblib
import time

# Load trained model
model = joblib.load("isl_knn_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# IMPORTANT FIX FOR WINDOWS
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

sentence = ""
prev_letter = ""
stable_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not reading frame")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    letter = ""

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        data = []
        for p in lm:
            data.append(p.x)
            data.append(p.y)

        letter = model.predict([data])[0]

        # Stability logic
        if letter == prev_letter:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= 15:
            sentence += letter
            stable_count = 0

        prev_letter = letter

    # Display text
    cv2.putText(frame, f"Letter: {letter}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Text: {sentence}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame, "SPACE = space | BACKSPACE = delete | Q = quit",
                (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # SHOW WINDOW (IMPORTANT)
    cv2.imshow("ISL Sentence Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:   # SPACE
        sentence += " "
        time.sleep(0.3)

    elif key == 8:  # BACKSPACE
        sentence = sentence[:-1]
        time.sleep(0.3)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
