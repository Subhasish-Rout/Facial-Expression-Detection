import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "expression_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "label_classes.npy")

# Standard 7 emotions fallback if classes file is not found
DEFAULT_CLASSES = np.array(["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"])

# Emotion color scheme (BGR format)
EMOTION_COLORS = {
    "happy": (0, 255, 0),        # Green
    "sad": (255, 100, 0),        # Blue
    "angry": (0, 0, 255),        # Red
    "surprise": (0, 255, 255),   # Yellow
    "fear": (180, 0, 180),       # Magenta
    "disgust": (0, 140, 255),    # Orange
    "neutral": (200, 200, 200)   # Light Gray
}

def load_resources():
    # 1. Load Haar Cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_classifier = cv2.CascadeClassifier(cascade_path)
    if face_classifier.empty():
        local_cascade = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
        if os.path.exists(local_cascade):
            face_classifier = cv2.CascadeClassifier(local_cascade)
        else:
            raise RuntimeError("Could not find Haar Cascade xml file.")

    # 2. Load Trained Model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. Run 'python train_model.py' first!"
        )
    print(f"Loading model from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)

    # 3. Load Class Names
    if os.path.exists(CLASSES_PATH):
        classes = np.load(CLASSES_PATH)
    else:
        classes = DEFAULT_CLASSES
    print(f"Classes loaded: {classes}")

    return face_classifier, model, classes

def draw_hud(frame, face_rect, label, confidence, all_preds, classes):
    """Draw stylish bounding box and probabilities HUD."""
    x, y, w, h = face_rect
    color = EMOTION_COLORS.get(label, (0, 255, 0))

    # Bounding Box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Corner Accents
    corner_length = int(min(w, h) * 0.15)
    thickness = 3
    # Top-Left
    cv2.line(frame, (x, y), (x + corner_length, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + corner_length), color, thickness)
    # Top-Right
    cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, thickness)
    # Bottom-Left
    cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, thickness)
    # Bottom-Right
    cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, thickness)

    # Header label pill
    label_text = f"{label.upper()} ({confidence:.1f}%)"
    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top_pos = max(y - 10, text_h + 10)
    cv2.rectangle(frame, (x, top_pos - text_h - 4), (x + text_w + 10, top_pos + baseline), color, -1)
    cv2.putText(
        frame,
        label_text,
        (x + 5, top_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )

    # Probability Bars Overlay on the Right/Side
    overlay_x = 10
    overlay_y = 30
    cv2.rectangle(frame, (overlay_x - 5, overlay_y - 20), (overlay_x + 210, overlay_y + (len(classes) * 22) + 10), (20, 20, 20), -1)
    cv2.putText(frame, "EMOTION PROBABILITIES", (overlay_x, overlay_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    for i, (cls_name, prob) in enumerate(zip(classes, all_preds)):
        row_y = overlay_y + 18 + (i * 22)
        bar_w = int(prob * 100)
        c = EMOTION_COLORS.get(cls_name, (200, 200, 200))
        # Text
        cv2.putText(frame, f"{cls_name[:7]:<7}", (overlay_x, row_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        # Background bar
        cv2.rectangle(frame, (overlay_x + 65, row_y), (overlay_x + 165, row_y + 12), (60, 60, 60), -1)
        # Active progress bar
        cv2.rectangle(frame, (overlay_x + 65, row_y), (overlay_x + 65 + bar_w, row_y + 12), c, -1)
        # Percentage text
        cv2.putText(frame, f"{prob*100:.0f}%", (overlay_x + 172, row_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

def main():
    print("=" * 50)
    print("Real-Time Facial Expression Detection")
    print("=" * 50)

    try:
        classifier, model, classes = load_resources()
    except Exception as e:
        print(f"\nInitialization Error: {e}")
        return

    print("\nSelect Camera Source:")
    print("1. Default Webcam (Index 0)")
    print("2. IP Webcam URL (e.g. http://192.168.1.X:8080/shot.jpg)")
    choice = input("Enter choice [1/2] (default 1): ").strip()

    is_ip = (choice == "2")
    url = ""
    cap = None

    if is_ip:
        default_url = "http://100.113.162.81:8080/shot.jpg"
        u = input(f"Enter IP URL (Press enter for {default_url}): ").strip()
        url = u if u else default_url
        import urllib.request
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

    print("\nStarting video stream. Press 'q' or 'ESC' to exit.")

    while True:
        if is_ip:
            try:
                req = urllib.request.urlopen(url, timeout=4)
                arr = np.array(bytearray(req.read()), dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except Exception:
                continue
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to grab camera frame.")
                break

        # Flip horizontally for natural mirror feel
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = classifier.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            # Crop face ROI
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (100, 100))
            roi_norm = roi_gray.astype("float32") / 255.0
            roi_input = np.expand_dims(roi_norm, axis=(0, -1))  # (1, 100, 100, 1)

            # Predict
            preds = model.predict(roi_input, verbose=0)[0]
            max_idx = int(np.argmax(preds))
            label = classes[max_idx]
            confidence = float(preds[max_idx] * 100)

            # Draw
            draw_hud(frame, (x, y, w, h), label, confidence, preds, classes)

        cv2.imshow("Facial Expression Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Camera stream closed.")

if __name__ == "__main__":
    main()

