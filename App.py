from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import datetime

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Load model and face detector
mymodel = load_model('mymodel.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            face = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in face:
                face_img = frame[y:y+h, x:x+w]
                cv2.imwrite('temp.jpg', face_img)
                test_image = image.load_img('temp.jpg', target_size=(150, 150, 3))
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis=0)
                pred = mymodel.predict(test_image)[0][0]

                if pred == 1:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    cv2.putText(frame, 'NO MASK', ((x+w)//2, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(frame, 'MASK', ((x+w)//2, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            datet = str(datetime.datetime.now())
            cv2.putText(frame, datet, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
