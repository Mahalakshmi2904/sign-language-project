from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3

app = Flask(__name__)

# Initialize speech engine
engine = pyttsx3.init()

# Load trained model
with open("isl_combined_model.pkl", "rb") as f:
    model = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

detected_char = ""


def generate_frames():

    global detected_char

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        detected_char = ""

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]

            lm = hand.landmark

            wrist = lm[0]

            data = []

            for p in lm:

                data.append(p.x - wrist.x)
                data.append(p.y - wrist.y)

            data = np.array(data)

            if np.max(np.abs(data)) != 0:

                data = data / np.max(np.abs(data))

            data = data.reshape(1, -1)

            prediction = model.predict(data)[0]

            confidence = np.max(model.predict_proba(data))

            if confidence > 0.75:

                detected_char = str(prediction)

            else:

                detected_char = ""

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

        cv2.putText(
            frame,
            f"Detected: {detected_char}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/detect")
def detect():

    return render_template("detect.html")


@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/get_char")
def get_char():

    global detected_char

    return jsonify({"char": detected_char})


@app.route("/speak", methods=["POST"])
def speak():

    data = request.get_json()

    if data:

        text = data.get("text", "")

        print("Speaking:", text)

        if text.strip() != "":

            engine = pyttsx3.init()

            engine.say(text)

            engine.runAndWait()

    return jsonify({"status": "spoken"})


if __name__ == "__main__":

    app.run(debug=True)