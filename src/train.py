#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, MobileNet, ResNet50  # Using ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))

# Parameters
#data_dir = "../dat/images_augmented_cropped" 
data_dir = "../dat/images"
target_size = (224, 224)
batch_size = 32
num_epochs = 1

validation_split = 0.2  # Percentage of data to use for validation

# Data Generator with Splitting
datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)

# Load data with the split
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training' 
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation' 
)

# Model Creation with Pre-trained Weights
def create_model(model_type):
    if model_type == "VGG16":
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=target_size + (3,))
    elif model_type == "MobileNet":
        base_model = MobileNet(include_top=False, weights='imagenet', input_shape=target_size + (3,))
    elif model_type == "ResNet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=target_size + (3,))
    elif model_type == "OwnModel":
        kernel_size =  3  # Example value
        max_pool =  2  # Example value
        base_model = Sequential([
            # First layer
            Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same', input_shape=target_size + (3,)),
            MaxPooling2D((max_pool, max_pool)),
            # Second layer
            Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same'),
            MaxPooling2D((max_pool, max_pool)),
            # Third layer
            Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same'),
            MaxPooling2D((max_pool, max_pool)),
            # Fourth layer
            Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same'),
            AveragePooling2D(pool_size=(2,  2), strides=(2,  2)),
            # Fifth layer
            Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same'),
            MaxPooling2D((max_pool, max_pool)),
])

    else:
        raise ValueError("Invalid model_type")

    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(train_data.num_classes, activation='softmax') 
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy", dtype=None), tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy", dtype=None)])
    return model

# Training and Evaluation
models = {
    "VGG16": create_model("VGG16"),
    "MobileNet": create_model("MobileNet"),
    "ResNet50": create_model("ResNet50"),
    "OwnModel": create_model("OwnModel"),
}
#%%
results = {}
for model_name, model in models.items():
    history = model.fit(train_data, epochs=num_epochs, validation_data=val_data, verbose=1)
    results[model_name] = history.history

    # Save model parameters
    model.save(f"../mod/{model_name}.h5")
    # Convert history to DataFrame
    df = pd.DataFrame(history.history)
    
    # Save DataFrame to CSV
    df.to_csv(f"{model_name}_history.csv", index=False)

print("Training and Evaluation Complete!")

# %%
