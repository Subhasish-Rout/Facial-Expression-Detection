import os
import cv2
import numpy as np
import pickle

# Resolve paths dynamically relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "clean_data")
IMG_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(DATA_DIR, exist_ok=True)

def preprocess(image, target_size=(100, 100)):
    """Convert to grayscale and resize."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, target_size)
    return image

def main():
    print("=" * 50)
    print("Consolidating & Preprocessing Dataset")
    print(f"Reading from: {IMG_DIR}")
    print(f"Saving to:    {DATA_DIR}")
    print("=" * 50)

    if not os.path.exists(IMG_DIR):
        print(f"Error: Images directory '{IMG_DIR}' does not exist.")
        return

    images = []
    labels = []
    skipped = 0

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_files = [f for f in os.listdir(IMG_DIR) if os.path.splitext(f.lower())[1] in valid_extensions]
    total_files = len(all_files)
    print(f"Total valid image files found: {total_files}")

    for idx, filename in enumerate(all_files, 1):
        file_path = os.path.join(IMG_DIR, filename)
        image = cv2.imread(file_path)
        
        if image is None:
            skipped += 1
            continue

        try:
            processed_img = preprocess(image, target_size=(100, 100))
            images.append(processed_img)
            # Label is the prefix before '_' (e.g. 'happy' from 'happy_ (1).jpg')
            label = filename.split('_')[0].strip().lower()
            labels.append(label)
        except Exception as e:
            skipped += 1
            continue

        if idx % 1000 == 0 or idx == total_files:
            print(f"Processed: {idx}/{total_files} images...")

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels)

    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nDataset Summary:")
    for lbl, count in zip(unique_labels, counts):
        print(f"  - {lbl}: {count} images")
    print(f"\nTotal loaded: {len(images)} images (Skipped/Corrupt: {skipped})")

    images_path = os.path.join(DATA_DIR, "images.p")
    labels_path = os.path.join(DATA_DIR, "labels.p")

    print(f"Saving {images_path}...")
    with open(images_path, "wb") as f:
        pickle.dump(images, f)

    print(f"Saving {labels_path}...")
    with open(labels_path, "wb") as f:
        pickle.dump(labels, f)

    print("Data consolidation complete! Ready for model training.")

if __name__ == "__main__":
    main()
