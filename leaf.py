# ===================================================
# 🌿 Leaf Disease Detection using CNN (PlantVillage)
# ===================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# -------------------------------
# 1. Dataset Path
# -------------------------------
# Change this path to your PlantVillage folder location
train_dir = r"C:\Users\Chinm\OneDrive\Desktop\codealpha\plantvillage"
val_dir = train_dir  # we’ll use the same folder with validation_split

# -------------------------------
# 2. Image Preprocessing & Data Augmentation
# -------------------------------
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,  # 80% training, 20% validation
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    subset='training',
    class_mode='categorical'  # ✅ Multi-class one-hot encoded
)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    subset='validation',
    class_mode='categorical'  # ✅ Must match training
)

print("✅ Classes found:", train_data.num_classes)
print("📂 Class indices:", train_data.class_indices)

# -------------------------------
# 3. Build CNN Model
# -------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')  # ✅ Output for all classes
])

# -------------------------------
# 4. Compile the Model
# -------------------------------
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',  # ✅ Must match class_mode='categorical'
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# 5. Train the Model
# -------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

# -------------------------------
# 6. Evaluate the Model
# -------------------------------
loss, accuracy = model.evaluate(val_data)
print(f"\n✅ Validation Accuracy: {accuracy * 100:.2f}%")

# -------------------------------
# 7. Plot Accuracy and Loss
# -------------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------------
# 8. Save Model
# -------------------------------
model.save("leaf_disease_detector.h5")
print("💾 Model saved as 'leaf_disease_detector.h5'")