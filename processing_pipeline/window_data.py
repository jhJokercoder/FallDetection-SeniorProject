import os
import pandas as pd
import numpy as np

# Get the current working directory 
cwd = os.getcwd()

# Get the parent directory
parent_dir = os.path.dirname(cwd)

# Get the absolute path to the data folder
data_folder_path = os.path.join(parent_dir, "data")

# Make a new folder for the Numpy files
windowed_data_folder_path = os.path.join(data_folder_path, "windowed_data")
os.makedirs(windowed_data_folder_path, exist_ok = True)

# Get the absolute path to the labeled_data folder
labeled_data_folder_path = os.path.join(data_folder_path, 'labeled_data')

# Get the list of subfolders in the labeled_data folder
labeled_data_subfolders_list = os.listdir(labeled_data_folder_path)

# The window size is 1 second which is equivalent to 100 rows in a CSV file from the dataset
window_size = 100

# These lists will store all the windows for the respective columns
acceleration_x_windows = []
acceleration_y_windows = []
acceleration_z_windows = []
gyroscope_x_windows = []
gyroscope_y_windows = []
gyroscope_z_windows = []
labels = []

for folder in labeled_data_subfolders_list:

	# Get the absolute path to the sensor data subfolder
	sensor_data_subfolder_path = os.path.join(labeled_data_folder_path, folder)

	# Get the list of files in the sensor data subfolder
	sensor_data_subfolder_files_list = os.listdir(sensor_data_subfolder_path)

	for data_file in sensor_data_subfolder_files_list:

		# Get the absolute file path for the sensor data file
		sd_file = os.path.join(sensor_data_subfolder_path, data_file)

		# Convert CSV file to Pandas Dataframe
		sd_dataframe = pd.read_csv(sd_file)

		# Find the number of rows in the Pandas Dataframe
		num_of_rows_sd_dataframe = len(sd_dataframe)

		# These lists will store the values of a window for the respective columns
		acc_x = []
		acc_y = []
		acc_z = []
		gyro_x = []
		gyro_y = []
		gyro_z = []
		label = []

		# Calculate the quotient and remainder of the number of rows of the dataframe and the window size
		quotient = num_of_rows_sd_dataframe // window_size
		remainder = num_of_rows_sd_dataframe % window_size

		# If the remainder is 0 then the length will be equal to the product of the window size and the quotient
		# If the remainder is not 0 then the length will be equal to the product of the window size and the value of quotient + 1
		if remainder == 0:
			length = window_size * quotient
		else:
			length = window_size * (quotient + 1)

		for i in range(length):
			
			# If the quotient of the current position and the window size does not produce a remainder
			# then append appropriate values for current position, append each window to respective window lists,
			# and clear the lists that store values for a window in order to start a new window.
			if (i+1) % window_size == 0:

				# If the current position is less than or equal to the number of rows of the dataframe 
				# then append the values of the columns to the respective lists that store the values for a window.
				if (i+1) <= (window_size * quotient) + remainder:
					acc_x.append(sd_dataframe.iloc[i, 2])
					acc_y.append(sd_dataframe.iloc[i, 3])
					acc_z.append(sd_dataframe.iloc[i, 4])
					gyro_x.append(sd_dataframe.iloc[i, 5])
					gyro_y.append(sd_dataframe.iloc[i, 6])
					gyro_z.append(sd_dataframe.iloc[i, 7])
					label.append(sd_dataframe.iloc[i, 11])

				# If the current position is greater than the number of rows of the dataframe then append 0's to 
				# the the lists that store the values for a window.
				elif (i+1) > (window_size * quotient) + remainder:
					acc_x.append(0)
					acc_y.append(0)
					acc_z.append(0)
					gyro_x.append(0)
					gyro_y.append(0)
					gyro_z.append(0)
					label.append(0)

				# Append each window to the respective window list, then clear the lists 
				# that store values for a window in order to start a new window.
				acceleration_x_windows.append(acc_x.copy())
				acc_x.clear()
				acceleration_y_windows.append(acc_y.copy())
				acc_y.clear()
				acceleration_z_windows.append(acc_z.copy())
				acc_z.clear()
				gyroscope_x_windows.append(gyro_x.copy())
				gyro_x.clear()
				gyroscope_y_windows.append(gyro_y.copy())
				gyro_y.clear()
				gyroscope_z_windows.append(gyro_z.copy())
				gyro_z.clear()

				# If 1 is in the label list then append 1 to the labels list.
				# If 1 is not in the label list then append 0 to the labels list. Next
				# clear the label list in order to start a new window.
				if 1 in label:
					labels.append(1)
				else:
					labels.append(0)

				label.clear()

			# If the quotient of the current position and the window size produces a remainder
			# then append appropriate values for the current position.
			elif (i+1) % window_size != 0:

				# If the current position is less than or equal to the number of rows of the dataframe 
				# then append the values of the columns to the respective lists that store the values for a window.
				if (i+1) <= (window_size * quotient) + remainder:
					acc_x.append(sd_dataframe.iloc[i, 2])
					acc_y.append(sd_dataframe.iloc[i, 3])
					acc_z.append(sd_dataframe.iloc[i, 4])
					gyro_x.append(sd_dataframe.iloc[i, 5])
					gyro_y.append(sd_dataframe.iloc[i, 6])
					gyro_z.append(sd_dataframe.iloc[i, 7])
					label.append(sd_dataframe.iloc[i, 11])

				# If the current position is greater than the number of rows of the dataframe then append 0's to 
				# the the lists that store the values for a window.
				elif (i+1) > (window_size * quotient) + remainder:
					acc_x.append(0)
					acc_y.append(0)
					acc_z.append(0)
					gyro_x.append(0)
					gyro_y.append(0)
					gyro_z.append(0)
					label.append(0)

# Put all the lists contatining the windows into one list
data = [acceleration_x_windows, acceleration_y_windows, acceleration_z_windows, 
        gyroscope_x_windows, gyroscope_y_windows, gyroscope_z_windows]

# Create numpy array using the data list
arr = np.array(data)
print("Windowed Data array shape:", arr.shape)
# Create numpy array using the labels list
arr2 = np.array(labels)
print("Labels array shape:", arr2.shape)

# Save the arrays to numpy files
np.save(os.path.join(windowed_data_folder_path, 'windowed_data.npy'), arr)
np.save(os.path.join(windowed_data_folder_path, 'labels.npy'), arr2)