import cv2 as cv
import numpy as np
import subprocess
import time

trigger_delay = 0.7   # seconds
extra_start_time = None

# LOAD MODEL 
net = cv.dnn.readNetFromCaffe(
    "face_dnn/deploy.prototxt",
    "face_dnn/res10_300x300_ssd_iter_140000.caffemodel"
)

# CAMERA
cap = cv.VideoCapture(0)

cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# ALLOWED KEYS
allowed_keys = {
    "ctrl+w": ["ctrl+w"],
    "alt+f4": ["alt+F4"],
    "ctrl+f": ["ctrl+f"],
    "ctrl+z": ["ctrl+z"],
    "ctrl+x": ["ctrl+x"],
    "ctrl+c": ["ctrl+c"],
    "ctrl+v": ["ctrl+v"],
    "ctrl+s": ["ctrl+s"],
    "ctrl+t": ["ctrl+t"]
}

# USER INPUT 
try:
    input_person = int(input("Type current person count: "))

    input_key = input(
        "Key to press automatically (example: ctrl+w): "
    ).lower()

    if input_key not in allowed_keys:
        print("Key not allowed.")
        exit()

except ValueError as e:
    print(e)
    exit()

# WINDOW SETUP 
window_name = "DNN Face Detection"

cv.namedWindow(window_name, cv.WINDOW_NORMAL)

# Give OpenCV time to create window
time.sleep(1)

# Minimize OpenCV window immediately
try:
    subprocess.run([
        "xdotool",
        "search",
        "--name",
        window_name,
        "windowminimize"
    ])
except Exception as e:
    print("Could not minimize window:", e)

# MAIN LOOP
triggered = False

while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    boxes = []
    confidences = []

    # CREATE BLOB 
    blob = cv.dnn.blobFromImage(
        frame,
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    # DETECTIONS
    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.3:

            box = detections[0, 0, i, 3:7] * np.array(
                [w, h, w, h]
            )

            (x1, y1, x2, y2) = box.astype("int")

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(confidence))

    # NMS
    indexes = cv.dnn.NMSBoxes(
        boxes,
        confidences,
        0.3,
        0.2
    )

    person_count = 0

    if len(indexes) > 0:

        person_count = len(indexes)

        for i in indexes.flatten():

            x, y, w_box, h_box = boxes[i]

            cv.rectangle(
                frame,
                (x, y),
                (x + w_box, y + h_box),
                (0, 255, 0),
                2
            )

            text = f"{confidences[i] * 100:.1f}%"

            cv.putText(
                frame,
                text,
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    # DISPLAY INFO
    cv.putText(
        frame,
        f"People: {person_count}",
        (20, 40),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # TRIGGER
    if person_count > input_person:

        # Start timer first time extra person appears
        if extra_start_time is None:
            extra_start_time = time.time()

        # Only trigger if stable long enough
        elapsed = time.time() - extra_start_time

        if elapsed >= trigger_delay:

            if not triggered:

                try:
                    subprocess.Popen([
                        "xdotool",
                        "key",
                        allowed_keys[input_key][0]
                    ])

                    print(
                        f"Triggered: {input_key} "
                        f"(Detected {person_count} people)"
                    )

                    triggered = True

                except Exception as e:
                    print("Trigger error:", e)

    else:
        extra_start_time = None
        triggered = False

    # SHOW WINDOW 
    cv.imshow(window_name, frame)

    # EXIT
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# CLEANUP 
cap.release()
cv.destroyAllWindows()
