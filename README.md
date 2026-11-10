# Facial Expression Detection System

A complete end-to-end Deep Learning system for real-time facial expression recognition using OpenCV and TensorFlow/Keras.

---

## 📁 Project Structure

```
facial_expression_detection/
├── images/                   # Dataset images categorized by emotion prefix
├── clean_data/               # Preprocessed binary pickle files (images.p, labels.p)
├── collect_data.py           # Capture custom face images via Webcam or IP Cam
├── consolidated.py           # Preprocess images to grayscale (100x100) & generate pickle files
├── train_model.py            # Train Deep CNN on dataset with Data Augmentation
├── detect_expression.py      # Real-time webcam facial expression detector with HUD
├── expression_model.keras    # Saved trained model (generated after training)
└── label_classes.npy         # Saved emotion class mapping
```

---

## 🚀 How to Run

### Step 1: Preprocess Dataset (Consolidate Data)
```bash
python consolidated.py
```
*Processes all 7,178 images into `clean_data/images.p` and `clean_data/labels.p`.*

---

### Step 2: Train the Deep CNN Model
```bash
python train_model.py
```
*Trains a multi-layer CNN with BatchNormalization, Dropout, and Data Augmentation across 7 emotions (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`). Saves `expression_model.keras` and `label_classes.npy`.*

---

### Step 3: Run Real-Time Detection (Webcam)
```bash
python detect_expression.py
```
*Features:*
- Detects faces with OpenCV Haar Cascade.
- Predicts expression and displays real-time probabilities for all 7 emotions.
- Supports both laptop webcam (option 1) and IP Webcam app (option 2).
- Press **'q'** or **'ESC'** to quit.

---

### Step 4 (Optional): Collect More Face Data
To add your own expressions to the dataset:
```bash
python collect_data.py
```

