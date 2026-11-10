import os
import cv2
import urllib.request
import numpy as np

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load Haar Cascade Classifier
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
classifier = cv2.CascadeClassifier(cascade_path)
if classifier.empty():
    # Fallback to local xml file if available
    local_cascade = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
    if os.path.exists(local_cascade):
        classifier = cv2.CascadeClassifier(local_cascade)
    else:
        raise RuntimeError("Failed to load Haarcascade classifier.")

def get_frame_from_url(url):
    """Fetch and decode an image frame from an IP camera URL."""
    try:
        req = urllib.request.urlopen(url, timeout=5)
        img_array = np.array(bytearray(req.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Error fetching frame from URL: {e}")
        return None

def main():
    print("=" * 50)
    print("Facial Expression Data Collection Tool")
    print("=" * 50)
    print("Select video source:")
    print("1. Default Webcam (Index 0)")
    print("2. IP Webcam URL (e.g. http://192.168.1.X:8080/shot.jpg)")
    source_choice = input("Enter choice [1/2] (default 1): ").strip()

    is_ip_cam = False
    url = ""
    cap = None

    if source_choice == "2":
        is_ip_cam = True
        default_url = "http://100.113.162.81:8080/shot.jpg"
        user_url = input(f"Enter IP Webcam URL (Press enter for {default_url}): ").strip()
        url = user_url if user_url else default_url
        print(f"Connecting to IP Camera at {url}...")
    else:
        cam_index = 0
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print(f"Error: Could not open webcam at index {cam_index}.")
            return
        print(f"Webcam {cam_index} opened successfully.")

    label = input("\nEnter expression label (e.g. angry, happy, sad, neutral, surprise, fear, disgust): ").strip().lower()
    if not label:
        label = "user"

    target_count = 100
    data = []
    print(f"\nCollecting {target_count} face samples for '{label}'.")
    print("Look at the camera with the specified facial expression.")
    print("Press 'ESC' in the preview window to exit early.\n")

    while len(data) < target_count:
        if is_ip_cam:
            frame = get_frame_from_url(url)
            if frame is None:
                continue
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to capture frame from webcam.")
                break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50)
        )

        preview_frame = frame.copy()

        # If faces are detected, crop and save the primary (largest) face
        if len(faces) > 0:
            # Sort faces by area descending to pick the most prominent face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]

            face_crop = frame[y:y+h, x:x+w].copy()
            data.append(face_crop)
            print(f"Captured: {len(data)}/{target_count}")

            # Draw green bounding box for captured face
            cv2.rectangle(preview_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                preview_frame,
                f"{label}: {len(data)}/{target_count}",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            cv2.imshow("Captured Face Crop", face_crop)

        # Draw status banner
        cv2.putText(
            preview_frame,
            f"Collecting '{label}' ({len(data)}/{target_count}) - ESC to exit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow("Preview Window", preview_frame)
        key = cv2.waitKey(25) & 0xFF
        if key == 27:  # ESC key
            print("Capture cancelled by user.")
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

    if len(data) == target_count:
        # Determine starting index to avoid overwriting existing samples
        existing_files = [f for f in os.listdir(IMAGE_DIR) if f.startswith(f"{label}_")]
        start_idx = len(existing_files) + 1

        print(f"Saving {len(data)} images to {IMAGE_DIR}...")
        for i, face_img in enumerate(data):
            file_name = f"{label}_ ({start_idx + i}).jpg"
            file_path = os.path.join(IMAGE_DIR, file_name)
            cv2.imwrite(file_path, face_img)
        print(f"Successfully saved {len(data)} images for '{label}'.")
    else:
        print(f"Insufficient data collected ({len(data)}/{target_count}). Images were not saved.")

if __name__ == "__main__":
    main()
