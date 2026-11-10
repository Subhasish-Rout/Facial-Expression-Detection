import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "clean_data")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "expression_model.keras")
CLASSES_SAVE_PATH = os.path.join(BASE_DIR, "label_classes.npy")

def load_data():
    images_path = os.path.join(DATA_DIR, "images.p")
    labels_path = os.path.join(DATA_DIR, "labels.p")

    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Preprocessed data not found in '{DATA_DIR}'. Run consolidated.py first!"
        )

    print(f"Loading data from {DATA_DIR}...")
    with open(images_path, "rb") as f:
        images = pickle.load(f)
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)

    # Normalize images to [0, 1] and add channel dimension (N, 100, 100, 1)
    images = images.astype("float32") / 255.0
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=-1)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    classes = label_encoder.classes_

    # Save class names for inference
    np.save(CLASSES_SAVE_PATH, classes)
    print(f"Classes ({len(classes)}): {classes}")

    # One-hot encode targets
    num_classes = len(classes)
    y = keras.utils.to_categorical(encoded_labels, num_classes=num_classes)

    return images, y, classes

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample to 48x48 for fast CPU computation and standard FER feature resolution
        layers.Resizing(48, 48),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),

        # Fully Connected
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    print("=" * 50)
    print("Training Facial Expression Recognition Model")
    print("=" * 50)

    X, y, classes = load_data()
    num_classes = len(classes)
    print(f"Total samples: {X.shape[0]}, Image shape: {X.shape[1:]}, Classes: {num_classes}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

    model = build_model(input_shape=X.shape[1:], num_classes=num_classes)
    model.summary()

    callbacks_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    epochs = 20
    batch_size = 64
    print(f"\nStarting training for {epochs} epochs (batch size: {batch_size})...")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )

    # Save final model
    model.save(MODEL_SAVE_PATH)
    print("\n" + "=" * 50)
    print(f"Training completed! Best model saved to: {MODEL_SAVE_PATH}")
    print(f"Class labels saved to: {CLASSES_SAVE_PATH}")
    print("=" * 50)

if __name__ == "__main__":
    main()
