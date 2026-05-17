import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "No hand"

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Fingers up/down
        index_up  = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_up   = lm[16].y < lm[14].y
        pinky_up  = lm[20].y < lm[18].y

        # DISTANCE-BASED thumb detection (KEY FIX)
        thumb_index_dist = distance(lm[4], lm[8])
        thumb_middle_dist = distance(lm[4], lm[12])

        # Gesture mapping
        if not index_up and not middle_up and not ring_up and not pinky_up:
            gesture = "A"

        elif index_up and not middle_up and not ring_up and not pinky_up and thumb_index_dist < 0.1:
            gesture = "D"

        elif index_up and not middle_up and not ring_up and not pinky_up and thumb_index_dist > 0.18:
            gesture = "L"

        else:
            gesture = "Unknown"

    cv2.putText(frame, f"Gesture: {gesture}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("MediaPipe Hand Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
