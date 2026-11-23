
import os
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

from src.features import (
    extract_hand_features,
    draw_hand_landmarks
)

from src.language import SpellAndSentenceHelper


MODEL_PATH = "models/sign_lstm.keras"
CLASSES_PATH = "models/classes.npy"
HAND_MODEL_PATH = "models/hand_landmarker.task"

# Check required files.
for path in [MODEL_PATH, CLASSES_PATH, HAND_MODEL_PATH]:
    if not os.path.exists(path):
        raise SystemExit(f"Required file missing: {path}")

# Load trained LSTM model.
model = tf.keras.models.load_model(MODEL_PATH)

classes = np.load(
    CLASSES_PATH,
    allow_pickle=True
)

helper = SpellAndSentenceHelper(
    "outputs/language_corpus.txt"
    if os.path.exists("outputs/language_corpus.txt")
    else None
)

# Configure MediaPipe Tasks.
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    detector.close()
    raise RuntimeError("Cannot open webcam.")

window = deque(maxlen=30)

text_tokens = []

last_label = None
stable = 0
last_committed = None

threshold = 0.80
timestamp_ms = 0

print("ISL REAL-TIME RECOGNITION STARTED")
print("Q = Quit | SPACE = Commit | C = Clear")

try:
    while True:

        ok, frame = cap.read()

        if not ok:
            print("Unable to read webcam frame.")
            break

        # Keep same orientation as dataset collection.
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms += 33

        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        landmarks = result.hand_landmarks

        draw_hand_landmarks(frame, landmarks)

        features = extract_hand_features(landmarks)

        window.append(features)

        label = "Collecting frames..."
        confidence = 0.0

        if len(window) == 30:

            sequence = np.expand_dims(
                np.asarray(window, dtype=np.float32),
                axis=0
            )

            probabilities = model.predict(
                sequence,
                verbose=0
            )[0]

            idx = int(np.argmax(probabilities))

            confidence = float(probabilities[idx])

            label = (
                str(classes[idx])
                if confidence >= threshold
                else "UNKNOWN"
            )

            if label == last_label and label != "UNKNOWN":
                stable += 1
            else:
                stable = 1
                last_label = label

            # Commit a stable prediction once.
            if (
                stable >= 12
                and label != last_committed
                and label != "UNKNOWN"
            ):
                text_tokens.append(label)
                last_committed = label

        current_text = " ".join(text_tokens)

        corrected = helper.correct_text(
            current_text.replace("_", " ")
        )

        # Display information.
        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 160),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            frame,
            f"Prediction: {label}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.1%}",
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Text: {corrected[-75:]}",
            (15, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "Q Quit | SPACE Commit | C Clear",
            (15, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1
        )

        cv2.imshow(
            "ISL Sign Language to Text",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("c"):
            text_tokens.clear()
            last_committed = None

        elif key == 32:
            if label not in (
                "UNKNOWN",
                "Collecting frames..."
            ):
                text_tokens.append(label)
                last_committed = label

finally:
    cap.release()
    detector.close()
    cv2.destroyAllWindows()

print("Recognized text:", " ".join(text_tokens))
