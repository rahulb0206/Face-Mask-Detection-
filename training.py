import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Data paths
dataset_path_with_mask = r"C:\Users\Rahul\Desktop\Face_Mask_Detection\dataset\with_mask\"
dataset_path_without_mask = r"C:\Users\Rahul\Desktop\Face_Mask_Detection\dataset\without_mask\"

# Initialize lists to store data and labels
data = []
labels = []

# Load images with mask
for img_file in os.listdir(dataset_path_with_mask):
    img_path = os.path.join(dataset_path_with_mask, img_file)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    data.append(image)
    labels.append(1)  

# Load images without mask
for img_file in os.listdir(dataset_path_without_mask):
    img_path = os.path.join(dataset_path_without_mask, img_file)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    data.append(image)
    labels.append(0)  

# Convert lists to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Data augmentation
augmentation = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Load MobileNetV2 base model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Construct the head of the model
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(1, activation="sigmoid")(head_model)

# Place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=base_model.input, outputs=head_model)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
opt = Adam(lr=1e-4, decay=1e-4 / 20)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model
history = model.fit(
    augmentation.flow(data, labels, batch_size=32),
    steps_per_epoch=len(data) // 32,
    epochs=20
)

# Save the trained model
model.save(r"C:\Users\Rahul\Desktop\Face_Mask_Detection\models\"mask_detector.model")
