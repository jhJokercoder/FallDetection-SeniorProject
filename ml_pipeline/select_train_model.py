import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow.keras import Input, Model, layers
from tensorflow.keras.layers import Concatenate

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Get the current working directory 
cwd = os.getcwd()

# Make a new folder for the logs
logs_folder_path = os.path.join(cwd, "logs")
os.makedirs(logs_folder_path, exist_ok = True)

# Make a new folder for the models
models_folder_path = os.path.join(cwd, "models")
os.makedirs(models_folder_path, exist_ok = True)

# Make a new folder for the plots
plots_folder_path = os.path.join(cwd, "plots")
os.makedirs(plots_folder_path, exist_ok = True)

# Experiment ID
experiment_ID = "008"

# Fallback if empty string is provided
if experiment_ID == "":
    experiment_ID = "default_experiment"

# Log file setup
log_filename = f"{experiment_ID}_log.txt"
log_path = os.path.join(logs_folder_path, log_filename)

# Open log file in write mode
log_file = open(log_path, "w")

# Tee class: duplicates console output to file + terminal
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

# Redirect stdout and stderr to both console and log file
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

print(f"Logging to: {log_path}")

# Get the parent directory
parent_dir = os.path.dirname(cwd)

# Get the absolute path to the data folder
data_folder_path = os.path.join(parent_dir, "data")

# Get the absolute path to the scalogram_data folder
scalogram_data_folder_path = os.path.join(data_folder_path, "scalogram_data")

# Get the path to the scalogram_data numpy file
arr_path = os.path.join(scalogram_data_folder_path, 'scalogram_data.npy')

# Get the path to the labels numpy file
arr_path2 = os.path.join(scalogram_data_folder_path, 'labels.npy')

# Load the numpy arrays and print their shapes and data types
arr = np.load(arr_path)
print("Scalogram Data Shape and Type:", arr.shape, arr.dtype)

labels = np.load(arr_path2)
print("Labels Shape and Type:", labels.shape, labels.dtype)

# Change the shape of the input data from (2, 42465, 3, 50, 100) to (42465, 2, 50, 100, 3)
arr = np.transpose(arr, (1, 0, 3, 4, 2))
print("Transposed Data Shape:", arr.shape)

# Split data into training (70%) and temporary (30%) sets using stratified sampling to maintain class balance across splits
x_train, x_temp, y_train, y_temp = train_test_split(
    arr,
    labels,
    test_size=0.30,
    stratify=labels,
    shuffle=True,
    random_state=42
)
print("Training/Temporary Split Shape:", x_train.shape, x_temp.shape, y_train.shape, y_temp.shape)

# Split temporary set equally into validation and test sets (15% each of the full dataset) using stratified sampling to maintain class distribution
x_val, x_test, y_val, y_test = train_test_split(
    x_temp,
    y_temp,
    test_size=0.5,
    stratify=y_temp,
    shuffle=True,
    random_state=42
)
print("Validation/Testing Split Shape:", x_val.shape, x_test.shape, y_val.shape, y_test.shape)

# Separate data into sensor modalities: acceleration (index 0) and gyroscope (index 1)
acc_train = x_train[:, 0]
gyro_train = x_train[:, 1]
print("Acc/Gyro Training Shape:", acc_train.shape, gyro_train.shape)

acc_val = x_val[:, 0]
gyro_val = x_val[:, 1]
print("Acc/Gyro Validation Shape:", acc_val.shape, gyro_val.shape)

acc_test = x_test[:, 0]
gyro_test = x_test[:, 1]
print("Acc/Gyro Testing Shape:", acc_test.shape, gyro_test.shape)

# Standardize acceleration and gyroscope data using training set mean and std (z-score normalization)
# The same statistics are applied to validation and test sets to avoid data leakage
acc_mean = acc_train.mean()
acc_std = acc_train.std()
print("Acc Mean/STD Shape:", acc_mean, acc_std)

gyro_mean = gyro_train.mean()
gyro_std = gyro_train.std()
print("Gyro Mean/STD Shape:", gyro_mean, gyro_std)

acc_train_norm = (acc_train - acc_mean) / (acc_std + 1e-8)
acc_val_norm = (acc_val - acc_mean) / (acc_std + 1e-8)
acc_test_norm = (acc_test - acc_mean) / (acc_std + 1e-8)
print("Acc Training/Validation/Testing Shape (Norm):", acc_train_norm.shape, acc_val_norm.shape, acc_test_norm.shape)

gyro_train_norm = (gyro_train - gyro_mean) / (gyro_std + 1e-8)
gyro_val_norm = (gyro_val - gyro_mean) / (gyro_std + 1e-8)
gyro_test_norm = (gyro_test - gyro_mean) / (gyro_std + 1e-8)
print("Gyro Training/Validation/Testing Shape (Norm):", gyro_train_norm.shape, gyro_val_norm.shape, gyro_test_norm.shape)

# Model Architecture

# Input shape
input_shape = (50, 100, 3)

# Stream 1
input1 = Input(shape=input_shape)
x1 = layers.Conv2D(32, (3, 3), padding='same')(input1)
x1 = layers.BatchNormalization()(x1)
x1 = layers.Activation('relu')(x1)
x1 = layers.MaxPooling2D((2, 2))(x1)
x1 = layers.Conv2D(32, (3, 3), padding='same')(x1)
x1 = layers.BatchNormalization()(x1)
x1 = layers.Activation('relu')(x1)
x1 = layers.GlobalAveragePooling2D()(x1)

# Stream 2
input2 = Input(shape=input_shape)
x2 = layers.Conv2D(32, (3, 3), padding='same')(input2)
x2 = layers.BatchNormalization()(x2)
x2 = layers.Activation('relu')(x2)
x2 = layers.MaxPooling2D((2, 2))(x2)
x2 = layers.Conv2D(32, (3, 3), padding='same')(x2)
x2 = layers.BatchNormalization()(x2)
x2 = layers.Activation('relu')(x2)
x2 = layers.GlobalAveragePooling2D()(x2)

# Merge output from two streams
merged = Concatenate()([x1, x2])

# Densely Connected Layers and Output Layer
x = layers.Dense(128, activation='relu')(merged)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=[input1, input2], outputs=output)

# Compile the model
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print the model summary
model.summary()

# Custom callback to track the learning rate
class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Log the learning rate at the end of each epoch
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        if not hasattr(self, 'lr_history'):
            self.lr_history = []
        self.lr_history.append(lr)

# Create the learning rate tracker
lr_tracker = LearningRateTracker()

# Define learning rate scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,        # Reduce the learning rate by half
    patience=3,        # Wait 3 epochs without improvement
    min_lr=1e-6,
    verbose=1
)

# Train the model
history = model.fit(
    [acc_train_norm, gyro_train_norm],
    y_train,
    validation_data=([acc_val_norm, gyro_val_norm], y_val),
    epochs=50,
    batch_size=32,
    callbacks=[lr_scheduler, lr_tracker]
)

# Save model with experiment ID
model_filename = f"{experiment_ID}_model.keras"
model.save(os.path.join(models_folder_path, model_filename))

# Extract history
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
lr_history = lr_tracker.lr_history

# Plot Accuracy and Learning Rate (scaled)
plt.figure(figsize=(10, 6))
plt.plot(train_acc, label='Training Accuracy', color='blue')
plt.plot(val_acc, label='Validation Accuracy', color='orange')

# Scale the learning rate to match the accuracy range (0 to 1)
scaled_lr_acc = [lr * max(train_acc) / max(lr_history) for lr in lr_history]
plt.plot(scaled_lr_acc, label='Learning Rate (scaled)', linestyle='--', color='green')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title(f'Training vs Validation Accuracy with Learning Rate ({experiment_ID})')
plt.legend(loc='lower left')
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
acc_plot_filename = f"{experiment_ID}_accuracy_lr.png"
plt.savefig(os.path.join(plots_folder_path, acc_plot_filename))
plt.close()

# Plot Loss and Learning Rate (scaled)
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss', color='blue')
plt.plot(val_loss, label='Validation Loss', color='orange')

# Scale the learning rate to match the loss range (0 to 0.5)
scaled_lr_loss = [lr * max(train_loss) / max(lr_history) for lr in lr_history]
plt.plot(scaled_lr_loss, label='Learning Rate (scaled)', linestyle='--', color='green')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Training vs Validation Loss with Learning Rate ({experiment_ID})')
plt.legend(loc='upper left')
plt.ylim(0, 0.5)
plt.grid(True)
plt.tight_layout()
loss_plot_filename = f"{experiment_ID}_loss_lr.png"
plt.savefig(os.path.join(plots_folder_path, loss_plot_filename))
plt.close()
