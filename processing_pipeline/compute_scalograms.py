import os
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Get the current working directory 
cwd = os.getcwd()

# Get the parent directory
parent_dir = os.path.dirname(cwd)

# Get the absolute path to the data folder
data_folder_path = os.path.join(parent_dir, "data")

# Make a new folder for the Numpy file containing the scalogram data
scalogram_data_folder_path = os.path.join(data_folder_path, "scalogram_data")
os.makedirs(scalogram_data_folder_path, exist_ok = True)

# Get the absolute path to the windowed_data folder
windowed_data_folder_path = os.path.join(data_folder_path, "windowed_data")

# Get the path to the windowed_data numpy file
arr_path = os.path.join(windowed_data_folder_path, 'windowed_data.npy')

# Get the path to the labels numpy file
arr_path2 = os.path.join(windowed_data_folder_path, 'labels.npy')

# Load the numpy arrays
arr = np.load(arr_path)
print("Windowed Data array shape:", arr.shape)

arr2 = np.load(arr_path2)
print("Labels array shape:", arr2.shape)

# These lists store the 3-channel arrays for their respective modality
acceleration_channel = []
gyroscope_channel = []

# Sampling frequency is 100 Hz since the change in time is 0.01 seconds so fs = (1/0.01) = 100 Hz
fs = 100

# Scales
scales = np.arange(1, 51)

for i in range(arr.shape[1]):
	
	# Get the ith acc_x window 
	signal1 = arr[0, i, :]

	# Get the ith acc_y window 
	signal2 = arr[1, i, :]

	# Get the ith acc_z window 
	signal3 = arr[2, i, :]

	# Get the ith gyr_x window 
	signal4 = arr[3, i, :]

	# Get the ith gyr_y window
	signal5 = arr[4, i, :]

	# Get the ith gyr_z window
	signal6 = arr[5, i, :]

	# Calculate the Continuous Wavelet Transform (CWT) of each signal using the Morlet wavelet.
	coeffs, freqs = pywt.cwt(signal1, scales, 'morl', sampling_period=1/fs)
	coeffs2, _ = pywt.cwt(signal2, scales, 'morl', sampling_period=1/fs)
	coeffs3, _ = pywt.cwt(signal3, scales, 'morl', sampling_period=1/fs)

	coeffs4, _ = pywt.cwt(signal4, scales, 'morl', sampling_period=1/fs)
	coeffs5, _ = pywt.cwt(signal5, scales, 'morl', sampling_period=1/fs)
	coeffs6, _ = pywt.cwt(signal6, scales, 'morl', sampling_period=1/fs)

	# Calculate magnitude of wavelet coefficients
	s1 = np.abs(coeffs)
	s2 = np.abs(coeffs2)
	s3 = np.abs(coeffs3)

	s4 = np.abs(coeffs4)
	s5 = np.abs(coeffs5)
	s6 = np.abs(coeffs6)

	# Stack numpy arrays along a new first axis to create a 3-channel arrays.
	# This is analogous to RGB images: each channel corresponds to one “color”
	rgb_acc = np.stack([s1, s2, s3], axis=0)
	rgb_gyro = np.stack([s4, s5, s6], axis=0)

	# Append the 3-channel array to appropriate lists that store the 3-channel arrays
	acceleration_channel.append(rgb_acc)
	gyroscope_channel.append(rgb_gyro)

# This list stores the collection of 3-channel arrays for acceleration and gyroscope data
data = [acceleration_channel, gyroscope_channel]

# Create numpy array using the data list
arr3 = np.array(data)
print("Scalogram Data array shape:", arr3.shape)

# Save the array to a numpy file
np.save(os.path.join(scalogram_data_folder_path, 'scalogram_data.npy'), arr3)
np.save(os.path.join(scalogram_data_folder_path, 'labels.npy'), arr2)


'''
# Plot scalogram
plt.figure(figsize=(8, 4))
plt.imshow(np.abs(coeffs), aspect='auto', origin='lower',
           extent=[0, len(signal)*dt, freqs[0], freqs[-1]])
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("CWT Scalogram")
plt.colorbar(label="|CWT|")
plt.show()
'''
