from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO
from collections import Counter
import os

app = Flask(__name__)
CORS(app)



# ---------------- MODEL ----------------

api_key = os.environ.get("ULTRALYTICS_API_KEY")
model = YOLO("yolov8n.pt", api_key=api_key)


# ---------------- UTILITIES ----------------
def pluralize(word, count):
    if count == 1:
        return word
    if word.endswith('y') and word[-2] not in 'aeiou':
        return word[:-1] + 'ies'
    elif word.endswith(('s', 'x', 'z', 'ch', 'sh')):
        return word + 'es'
    else:
        return word + 's'


def detect_objects(frame, descriptive=False):
    results = model(frame, verbose=False)
    detected = []

    if results[0].boxes is not None:
        detected = [model.names[int(box.cls)] for box in results[0].boxes]

    if detected:
        counter = Counter(detected)
        desc = [f"{count} {pluralize(name, count)}" for name, count in counter.items()]
        if descriptive:
            return "In your environment, I see " + ", ".join(desc)
        return "I see " + ", ".join(desc)
    return "I don't see any recognizable objects"


# ---------------- ENVIRONMENT SCAN ----------------
env_scan_active = False
env_scan_frame = None
latest_scan_result = ""
scan_lock = threading.Lock()

def environment_scan_loop():
    global env_scan_active, env_scan_frame, latest_scan_result
    last_announcement_time = 0
    while True:
        time.sleep(1)
        if env_scan_active and env_scan_frame is not None:
            current_time = time.time()
            if last_announcement_time == 0 or current_time - last_announcement_time >= 10:
                with scan_lock:
                    latest_scan_result = detect_objects(env_scan_frame, descriptive=True)
                    print("[ENV SCAN]", latest_scan_result)
                last_announcement_time = current_time

def look_around_periodic(frame):
    global env_scan_frame
    with scan_lock:
        env_scan_frame = frame.copy()
    threading.Thread(target=environment_scan_loop, daemon=True).start()

# ---------------- ROUTES ----------------
@app.route('/process_frame', methods=['POST'])
def process_frame():
    global env_scan_active, latest_scan_result

    data = request.json
    image_b64 = data.get('frame', '')
    cmd = data.get('cmd', '').lower()

    if ',' in image_b64:
        image_b64 = image_b64.split(',')[1]

    img_bytes = base64.b64decode(image_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"response": "Invalid image"})

    if "what's there" in cmd:
        response_text = detect_objects(frame)
    elif "look around" in cmd:
        env_scan_active = True
        look_around_periodic(frame)
        response_text = "Environment scanning started."
    elif "pause vision" in cmd or "stop" in cmd:
        env_scan_active = False
        response_text = "Environment detection paused."
    else:
        response_text = "Unknown YOLO command."

    return jsonify({"response": response_text})


@app.route('/get_scan_result', methods=['GET'])
def get_scan_result():
    global latest_scan_result
    with scan_lock:
        result = latest_scan_result
        latest_scan_result = ""
    return jsonify({"latest_scan": result})


@app.route('/health')
def health():
    return jsonify({"status": "yolo server running"})


if __name__ == "__main__":
    print("=" * 50)
    print("🚀 YOLO Detection Server Running...")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000)
