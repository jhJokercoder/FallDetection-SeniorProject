import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, precision_recall_curve, 
                             confusion_matrix, ConfusionMatrixDisplay, classification_report)

# Get the current working directory 
cwd = os.getcwd()

# Get the parent directory
parent_dir = os.path.dirname(cwd)

# Make a new folder for the classification reports
classification_reports_folder_path = os.path.join(cwd, "classification_reports")
os.makedirs(classification_reports_folder_path, exist_ok = True)

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

# Get the absolute path to the models folder
models_folder_path = os.path.join(cwd, "models")

#Get the absolute path to the plots folder
plots_folder_path = os.path.join(cwd, "plots")

# Find all model files matching the pattern *_model.keras
model_files = [f for f in os.listdir(models_folder_path) if f.endswith('_model.keras') and f[:3].isdigit()]

# Sort model files by the 3-digit experiment ID numerically
model_files.sort(key=lambda f: int(f[:3]))  # Sort by the first 3 digits as integers

# Prepare to store the PR AUC, Precision-Recall curve data, confusion matrices, and classification reports
pr_auc_values = []
precision_values = []
recall_values = []
model_labels = []
confusion_matrices = []
classification_reports = []

# Loop over each sorted model file
for model_file in model_files:
    # Get the absolute path to the model file
    model_file_path = os.path.join(models_folder_path, model_file)
    
    # Load the model
    model = load_model(model_file_path)

    # Predict probabilities for the current model
    y_pred_probs = model.predict([acc_test_norm, gyro_test_norm])
    
    # Threshold predictions
    y_pred = (y_pred_probs >= 0.5).astype(int).flatten()

    # Calculate PR AUC for the current model
    pr_auc = average_precision_score(y_test, y_pred_probs)
    
    # Calculate Precision-Recall curve values
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_probs)
    
    # Calculate Confusion Matrix for the current model
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate Classification Report for the current model
    class_report = classification_report(y_test, y_pred)
    
    # Store the results
    pr_auc_values.append(pr_auc)
    precision_values.append(precision_vals)
    recall_values.append(recall_vals)
    confusion_matrices.append(cm)
    classification_reports.append(class_report)
    
    # Label each model with its experiment ID (the first 3 digits of the filename)
    model_labels.append(model_file[:3])

# Plot all Precision-Recall curves on the same plot
plt.figure(figsize=(8, 6))
for i, label in enumerate(model_labels):
    plt.plot(recall_values[i], precision_values[i], label=f'{label} (PR AUC = {pr_auc_values[i]:.3f})')

# Create the x-axis annd y-axis ticks with a step of 0.1 from 0.1 to 1.0
ticks = np.arange(0.0, 1.1, 0.1)
plt.xticks(ticks)
plt.yticks(ticks)

# Add plot labels, title, and grid
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for All Models")
plt.grid(True)
plt.legend(loc='best')

# Save the combined Precision-Recall Curve plot
plt.savefig(os.path.join(plots_folder_path, "combined_precision_recall_curve.png"))
plt.close()

# Print the PR AUC for each model
for i, label in enumerate(model_labels):
    print(f"Experiment {label}: PR AUC = {pr_auc_values[i]:.4f}")

# Plot and save confusion matrices for each model
for i, cm in enumerate(confusion_matrices):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues, values_format='d')  # 'd' stands for decimal format (integers)
    plt.title(f"Confusion Matrix ({model_labels[i]})")
    plt.savefig(os.path.join(plots_folder_path, f"{model_labels[i]}_confusion_matrix.png"))
    plt.close()
    print(f"Confusion Matrix for Experiment {model_labels[i]}:\n{cm}\n")

# Print or save the classification reports
for i, report in enumerate(classification_reports):
    print(f"Classification Report for Experiment {model_labels[i]}:\n{report}\n")
    # Save the classification report to a text file
    with open(os.path.join(classification_reports_folder_path, f"{model_labels[i]}_classification_report.txt"), 'w') as f:
        f.write(report)