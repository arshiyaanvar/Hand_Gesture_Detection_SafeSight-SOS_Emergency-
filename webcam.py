import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from collections import deque

# Load Model
model = load_model("model/gesture_model.h5")

# Classes (order MUST match model.class_indices)
CLASS_NAMES = ["fist", "no_gesture", "palm"]

# Colors
COLOR_FIST = (0, 0, 255)        # Red
COLOR_NO = (0, 255, 255)        # Yellow
COLOR_PALM = (0, 255, 0)        # Green

# SOS variables
fist_start_time = None
previous_label = "none"
SOS_HOLD_TIME = 3               # seconds required to hold fist
sos_triggered = False
SOS_COOLDOWN = 4
last_sos_time = 0

# Smoothing window
smooth = deque(maxlen=4)

# Preprocess frame
def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict
    img = preprocess(frame)
    preds = model.predict(img, verbose=0)[0]
    class_id = np.argmax(preds)
    label = CLASS_NAMES[class_id]

    # Smooth predictions
    smooth.append(label)
    label = max(set(smooth), key=smooth.count)

    color = (255, 255, 255)

    # ============================
    # Gesture Detection Logic
    # ============================
    if label == "palm":
        color = COLOR_PALM
        fist_start_time = None
        sos_triggered = False

    elif label == "no_gesture":
        color = COLOR_NO
        fist_start_time = None
        sos_triggered = False

    elif label == "fist":
        color = COLOR_FIST

        # Detect palm → fist transition
        if previous_label == "palm" and fist_start_time is None:
            fist_start_time = time.time()

        # If fist is held
        if fist_start_time is not None:
            elapsed = time.time() - fist_start_time

            # If held for required time
            if elapsed >= SOS_HOLD_TIME and not sos_triggered:
                if time.time() - last_sos_time >= SOS_COOLDOWN:

                    sos_triggered = True
                    last_sos_time = time.time()

                    print("🚨 EMERGENCY SOS TRIGGERED! 🚨")

    previous_label = label

    # ============================
    # DISPLAY LABELS
    # ============================
    if sos_triggered:
        cv2.putText(frame, "SOS TRIGGERED!", (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)
    else:
        cv2.putText(frame, label.upper(), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("SafeSight Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()