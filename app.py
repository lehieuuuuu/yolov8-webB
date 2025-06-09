from flask import Flask, render_template, Response, request
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)
model = YOLO("good.pt")


cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model.predict(frame, conf=0.25)
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_image = None
    if request.method == 'POST':
        # File input name trong form bạn là 'image', lấy file upload
        file = request.files.get('image')
        if file:
            # Đọc file thành ảnh OpenCV
            img_bytes = file.read()
            npimg = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            # Dùng YOLO predict
            results = model.predict(img, conf=0.25)
            annotated_img = results[0].plot()

            # Chuyển ảnh sang base64 để nhúng trực tiếp vào HTML
            _, buffer = cv2.imencode('.jpg', annotated_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            uploaded_image = img_base64

    return render_template('index.html', uploaded_image=uploaded_image)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
