import cv2
import mediapipe as mp
import joblib

# Load trained alphabet model
model = joblib.load("isl_knn_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    predicted_letter = "No Hand"

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        data = []
        for p in lm:
            data.append(p.x)
            data.append(p.y)

        predicted_letter = model.predict([data])[0]

    cv2.putText(frame, f"Detected: {predicted_letter}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 3)

    cv2.imshow("Alphabet Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
