from flask import Flask
from flask import render_template 
from flask import Response 

import cv2 

app = Flask(__name__)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_detection = cv2.CascadeClassifier(cv2.data.haarcascades +
"haarcascade_frontalface_default.xml")

def function():
    while True:
        img, marco = cap.read()
        if img:
            gray = cv2.cvtColor(marco, cv2.COLOR_BGR2GRAY)
            faces = face_detection.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(marco, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.imshow('marco', marco)
            (flag, encodedImage) = cv2.imencode(".jpg", marco)
            if not flag:
                continue
            yield(b'--marco\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                 bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(function(),
    mimetype = "multipart/x-mixed-replace; boundary=marco")

if __name__ == "__main__":
    app.run(debug=False)

cap.release()