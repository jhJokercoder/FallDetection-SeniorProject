import os
import pandas as pd

# Get the current working directory 
cwd = os.getcwd()

# Get the parent directory
parent_dir = os.path.dirname(cwd)

# Get the absolute path to the data folder
data_folder_path = os.path.join(parent_dir, "data")

# Make a new folder for the labeled data
labeled_data_folder_path = os.path.join(data_folder_path, "labeled_data")
os.makedirs(labeled_data_folder_path, exist_ok = True)

# Get the absolute path to the KFall Dataset folder, the label_data folder, and the sensor_data folder
kfall_folder_path = os.path.join(data_folder_path, "KFall Dataset")
labels_folder_path = os.path.join(kfall_folder_path, "label_data")
sensor_data_folder_path = os.path.join(kfall_folder_path, "sensor_data")

# Get the list of files in the label_data folder
label_data_files_list = os.listdir(labels_folder_path)

for file in label_data_files_list:

	# Get the sensor_data subfolder name and Subject ID that corresponds to the current label file
	sensor_data_subfolder = file.split('_')[0]
	subject_id = sensor_data_subfolder.split('SA')[1]

	# Get the absolute path to the sensor_data subfolder
	sensor_data_subfolder_path = os.path.join(sensor_data_folder_path, sensor_data_subfolder)

	# Get the list of files in the sensor_data subfolder
	sensor_data_subfolder_files_list = os.listdir(sensor_data_subfolder_path)

	# Get the unique task id's from the file names in the sensor_data subfolder
	task_ids = []
	for sensor_file in sensor_data_subfolder_files_list:
		split_filename = sensor_file.split('T')[1]
		tsk_id = split_filename.split('R')[0]

		if tsk_id in task_ids:
			pass
		else:
			task_ids.append(tsk_id)

	# Get the absolute file path for the current label file
	label_file = os.path.join(labels_folder_path, file)

	# Convert Excel file to Pandas Dataframe
	label_dataframe = pd.read_excel(label_file)

	# Find the number of rows in the Pandas Dataframe
	num_of_rows_label_dataframe = len(label_dataframe)

	current_task_id = ''
	for i in range(num_of_rows_label_dataframe):

		# Get the Task Code (Task ID)
		task_code_task_id = label_dataframe.at[i, "Task Code (Task ID)"]

		# If the Task Code (Task ID) column has a null value then keep the current Task ID, if not then get the new Task ID
		if pd.isnull(task_code_task_id):
			task_id = current_task_id
		else:

			# Get the Task Id from the Task Code (Task Id) column, make it the new current Task ID, and remove it from the task_ids list
			first_split = task_code_task_id.split('(')[1]
			task_id = first_split.split(')')[0]
			current_task_id = task_id
			task_ids.remove(task_id)

		# Get the Trial ID 
		trial_id_num = label_dataframe.at[i, "Trial ID"]

		# Convert the Trial ID to a string and add 0 in front if the Trial ID is less than 10
		if trial_id_num	< 10:
			trial_id = '0' + str(trial_id_num)
		else:
			trial_id = str(trial_id_num)

		# Get the sensor_data filename
		sensor_data_filename = 'S' + subject_id + 'T' + task_id + 'R' + trial_id + '.csv'

		# Get the sensor_data file path
		sensor_data_file_path = os.path.join(sensor_data_subfolder_path, sensor_data_filename)

		# Convert CSV file to Pandas Dataframe
		sensor_dataframe = pd.read_csv(sensor_data_file_path)

		# Find the number of rows in the Pandas Dataframe
		num_of_rows_sensor_dataframe = len(sensor_dataframe)

		# Get the Onset Frame and Impact Frame
		onset_frame = label_dataframe.at[i, "Fall_onset_frame"]
		impact_frame = label_dataframe.at[i, "Fall_impact_frame"]

		# Create the Label column
		labels = []
		for i in range(num_of_rows_sensor_dataframe):
			#If the value in the FrameCounter column is between the onset_frame and impact_frame, inclusive, then append 1 to the labels list, if not then append 0
			if sensor_dataframe.at[i, "FrameCounter"] >= onset_frame and sensor_dataframe.at[i, "FrameCounter"] <= impact_frame:
				labels.append(1)
			else:
				labels.append(0)

		# Create Label column in the Pandas Dataframe
		sensor_dataframe['Label'] = labels

		# Make a new subfolder path in the labeled_data folder, then make the new labeled csv file and add it to the subfolder in the labeled_data folder
		labeled_data_subfolder_path = os.path.join(labeled_data_folder_path, sensor_data_subfolder)
		os.makedirs(labeled_data_subfolder_path, exist_ok = True)
		new_csv = sensor_dataframe.to_csv(os.path.join(labeled_data_subfolder_path, sensor_data_filename), index = False)

	# For the remaining Task Ids label all the files in the subfolder that have one of the remaining Task Ids
	for i in task_ids:
		t_id = 'T' + i

		for sensor_file in sensor_data_subfolder_files_list:

			# Get the Task Id from the filename
			split_filename = sensor_file.split('T')[1]
			id_num = split_filename.split('R')[0]
			compare_task_id = 'T' + id_num

			# If the Task Id in the filename is the same as the current Task Id then create labels that are all 0's, if not then pass
			if t_id == compare_task_id:

				# Convert CSV file to Pandas Dataframe
				sensor_dataframe2 = pd.read_csv(os.path.join(sensor_data_subfolder_path, sensor_file))

				# Find the number of rows in the Pandas Dataframe
				num_of_rows = len(sensor_dataframe2)

				# Create the Label column
				labels2 = []

				for i in range(num_of_rows):
					# Append 0 to the labels2 list
					labels2.append(0)

				# Create the Label column in the Pandas Dataframe
				sensor_dataframe2['Label'] = labels2

				# Make the new labeled csv file and add it to the new subfolder in the labeled_data folder
				new_csv2 = sensor_dataframe2.to_csv(os.path.join(labeled_data_subfolder_path, sensor_file), index = False)
			else:
				pass