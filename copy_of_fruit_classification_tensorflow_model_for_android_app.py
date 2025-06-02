import os
import urllib.request
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Download and unzip dataset
url = "https://bitbucket.org/ishaanjav/code-and-deploy-custom-tensorflow-lite-model/raw/a4febbfee178324b2083e322cdead7465d6fdf95/fruits.zip"
zip_path = "fruits.zip"
if not os.path.exists("fruits"):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Dataset extracted.")

# Load datasets
image_height, image_width = 32, 32
batch_size = 20

train_ds = tf.keras.utils.image_dataset_from_directory(
    "fruits/train",
    image_size=(image_height, image_width),
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "fruits/validation",
    image_size=(image_height, image_width),
    batch_size=batch_size
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "fruits/test",
    image_size=(image_height, image_width),
    batch_size=batch_size
)

class_name = ["apple", "banana", "orang"]

# Show sample images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_name[labels[i]])
        plt.axis("off")
plt.savefig("sample_images.png")  # Save instead of showing for headless environments

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="softmax"),
    tf.keras.layers.Dense(3)
])

model.compile(
    optimizer="rmsprop",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Train model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

# Evaluate and visualize predictions
plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
    predictions = model(images)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        pred_index = np.argmax(predictions[i])
        plt.title(f"Pred: {class_name[pred_index]} | Real: {class_name[labels[i]]}")
        plt.axis("off")
plt.savefig("test_predictions.png")  # Save instead of showing

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as model.tflite")
