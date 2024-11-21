import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
base_dir = "data" # Replace with your dataset path
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
img_height, img_width = 150, 150
batch_size = 32
train_datagen = ImageDataGenerator(
rescale=1.0/255,
rotation_range=30,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode="nearest"
)
val_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(img_height, img_width),
batch_size=batch_size,
class_mode="binary" # Binary classification
)
val_generator = val_datagen.flow_from_directory(
val_dir,
target_size=(img_height, img_width),
batch_size=batch_size,
class_mode="binary"
)
# Build the CNN model
model = Sequential([
Conv2D(32, (3, 3), activation="relu", input_shape=(img_height, img_width, 3)),
MaxPooling2D(pool_size=(2, 2)),
Conv2D(64, (3, 3), activation="relu"),
MaxPooling2D(pool_size=(2, 2)),
Conv2D(128, (3, 3), activation="relu"),
MaxPooling2D(pool_size=(2, 2)),
Flatten(),
Dense(128, activation="relu"),
Dropout(0.5),
Dense(1, activation="sigmoid") # Sigmoid for binary classification
])
model.compile(optimizer="adam",
loss="binary_crossentropy",
metrics=["accuracy"])
model.summary()
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
epochs = 25
history = model.fit(
train_generator,
steps_per_epoch=train_generator.samples // batch_size,
validation_data=val_generator,
validation_steps=val_generator.samples // batch_size,
epochs=epochs,
callbacks=[early_stopping]
)
model.save("organic_recyclable_classifier.h5")
print("Model saved!")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Accuracy Over Epochs")
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Loss Over Epochs")
plt.show()

