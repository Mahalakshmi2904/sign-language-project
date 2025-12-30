from flask import Flask, render_template
import cv2

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/start-camera")
def start_camera():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Camera - Press Q to Exit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "Camera closed"

if __name__ == "__main__":
    app.run(debug=True)
