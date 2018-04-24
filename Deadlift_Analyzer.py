#imports
import scipy.io
import numpy as np
import pandas as pd
from numpy import array
import csv
import sys
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from pathlib import Path
from matplotlib import pyplot as plt
from keras.utils import to_categorical
np.set_printoptions(threshold=sys.maxsize)

np.random.seed(123)  # for reproducibility

# def transform_kinect_files_to_csv():
#     raw_data_directory  = f'Skeleton_Data/raw_data/'
#     process_data_directory =  f'Skeleton_Data/processed_data/'
#     for filename in os.listdir(raw_data_directory):
#         if "DS_Store" not in filename:
#             if "b" in filename:
#                 processed_filename = filename.replace("b_","a11_s1_t")
#             else:
#                 processed_filename = filename.replace("g_","a1_s1_t")
#
#             trial_string = processed_filename[-2:]
#             trial_string = trial_string. replace("t","")
#             trial_string = trial_string.replace("_","")
#             trial_count = float(trial_string)
#             processed_filename += "_skeleton"
#             f= open(process_data_directory + "/"+ processed_filename + ".csv","w+")
#             f.write("action,subject,trial,frame,skeleton_joint,x,y,z\n")
#             with open('{}/{}'.format(raw_data_directory, filename)) as current_file_f:
#                 current_file = current_file_f.readlines()
#                 frame_count = 0
#                 for frame in current_file[:-1]:
#                     joint_count = 1
#                     for xyz_coords_str in frame.replace(";\n","").split(';'):
#                         xyz_coords = xyz_coords_str.split(',')
#                         try:
#                             x = float(xyz_coords[0]);
#                             y = float(xyz_coords[1]);
#                             z = float(xyz_coords[2]);
#
#                             if "a11" in processed_filename:
#                                 output = '11.0, 1.0, ' + str(trial_count) + ', ' + str(frame_count) + ', ' + str(joint_count) + ',' + str(x) + ', ' + str(y) +', ' + str(z) + '\n'
#                                 f.write(output)
#                             else:
#                                 output = '1.0, 1.0, ' + str(trial_count) + ', ' + str(frame_count) + ', ' + str(joint_count) + ',' + str(x) + ', ' + str(y) +', ' + str(z) + '\n'
#                                 f.write(output)
#                         except:
#                             continue
#                         #     print(filename)
#                         #     print(xyz_coords)
#                         #     exit()
#                         joint_count = joint_count + 1;
#                     frame_count = frame_count + 1;
#
#         else:
#             continue

# def transform_csv_to_mat():
#     directory =  f'Skeleton_Data/processed_data/'
#     for filename in os.listdir(directory):
#         if "DS_Store" not in filename:
#             csv_file_path = directory + filename
#             mat_file_path = csv_file_path[:-4] + '.mat'
#
#             # read first line of csv file to get column names and replace ' ' with '_'
#             r = csv.reader(csv_file_path)
#             header = next(r)
#             names = [x.replace(' ', '_') for x in header]
#             print(header)
#             print(csv_file_path)
#
#             # load the data into a pandas dataframe using the appropriate names
#             df = pd.read_csv(csv_file_path, names=names, header=0, skiprows=1, skipfooter=1)
#
#             # write the dataframe to a mat file
#             df_dict = {c: list(df[c]) for c in df.columns}
#             scipy.io.savemat(mat_file_path, df_dict, oned_as='column')
#


def import_skeleton_data(action, subject, trial):
    filename = f'Skeleton_Data/matrix_numpy_data/a{action}_s{subject}_t{trial}_skeleton.npy'
    if Path(filename).is_file():
        mat = np.load(filename)
        # print(mat)
        # exit()
        return mat
    else:
        return None

def transform_skeleton_data(action, subject, trial):
    matrices = []
    data = import_skeleton_data(action, subject, trial)
    if data is None: return None
    for frame in range(data.shape[2]):
        skelecton_joints = [i + 1 for i in range(16)] #number of joints
        matrix = data[:,:,frame]
        matrix = np.insert(matrix, 0, skelecton_joints, axis=1)
        matrix = np.insert(matrix, 0, frame, axis=1)
        matrices.append(matrix)
    result = np.vstack(tuple(matrices))
    result = np.insert(result, 0, [[action], [subject], [trial]], axis=1)
    return result

def transform_skeleton_data_to_df(action, subject, trial):
    data = transform_skeleton_data(action, subject, trial)
    if data is None: return None
    df = pd.DataFrame(data)
    df.columns = ['action', 'subject', 'trial', 'frame', 'skeleton_joint', 'x', 'y', 'z']
    return df


X_train = []
Y_train = []
X_test = []
Y_test = []


#     activites for training:
#     1.  Conventional Correct
#     11. Conventional Incorrect
#
activities = [1, 11]

for index, action in enumerate(activities):
    for subject in range(0, 9): #n-1 subjects
        for trial in range(0, 4): #n-1 trial
            data = import_skeleton_data(action, subject, trial)
            data = transform_skeleton_data(action, subject, trial)
            #export_inertial_data_to_csv(action, subject, trial)
            if data is None: continue
            data = np.swapaxes(data, 0, 1)
            data = sequence.pad_sequences(data, maxlen=326)
            if subject in [0, 1, 2, 4, 5, 6, 8 ] : #subjects used for test data [0, 1, 2, 4, 5, 6, 8 ]
                X_train.append(data)
                Y_train.append(index)

            else:
                X_test.append(data)
                Y_test.append(index)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
print(X_train)
print('X_train.shape:', X_train.shape)
print('Y_train.shape:', Y_train.shape)
print('X_test.shape:', X_test.shape)
print('Y_test.shape:', Y_test.shape)

# Swap axes again, new dimension is 32 x 326 x 6
# This follows the standard of LSTM: Samples, Timesteps, Dimensions
X_train = np.swapaxes(X_train, 1, 2)
X_test = np.swapaxes(X_test, 1, 2)

print('X_train.shape:', X_train.shape)
print('Y_train.shape:', Y_train.shape)
print('X_test.shape:', X_test.shape)
print('Y_test.shape:', Y_test.shape)



# One hot encoding
Y_train = to_categorical(Y_train, num_classes=len(activities))
Y_test = to_categorical(Y_test, num_classes=len(activities))

print('X_train.shape:', X_train.shape)
print('Y_train.shape:', Y_train.shape)
print('X_test.shape:', X_test.shape)
print('Y_test.shape:', Y_test.shape)

# Create the model
np.random.seed(7)
model = Sequential()
model.add(LSTM(200, input_shape=(326, 8)))
model.add(Dense(len(activities), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
print(model.summary())

# Train model
history = model.fit(X_train, Y_train, epochs=10, batch_size=1)

# Evaluate model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save(f'DeadLift_Model.h5')

# to make nice graph
import seaborn as sns
sns.set(style="darkgrid")
plt.plot(history.history['acc'])
plt.title('RNN LSTM - Activity Classification')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()


# def import_skeleton_x_train_data():
#     directory  = f'data/train/'
#     # path, dirs, files = os.walk(directory).next()
#     # file_count = len(files)
#
#     all_data = []
#     for filename in os.listdir(directory):
#         if "form" in filename:
#
#             temp_files = []
#             with open('{}/{}'.format(directory, filename)) as current_file_f:
#                 temp_frames = []
#
#                 current_file = current_file_f.readlines()
#
#                 for frame in current_file[:-1]:
#
#                     temp_joints = []
#
#                     for xyz_coords_str in frame.replace(";\n","").split(';'):
#                         xyz_coords = xyz_coords_str.split(',')
#                         try:
#                             x = float(xyz_coords[0]);
#                             y = float(xyz_coords[1]);
#                             z = float(xyz_coords[2]);
#                         except:
#                             print(filename)
#                             print(xyz_coords)
#                             exit()
#
#                         temp_joints.append([x, y, z])
#                     temp_frames.append(temp_joints)
#                 temp_files.append(temp_frames)
#             all_data.append(temp_files)
#         else:
#             continue
#
#     return all_data

# def import_skeleton_y_train_data():
#     directory  = f'data/train/'
#     output= []
#     for filename in os.listdir(directory):
#         if "good" in filename:
#             output.append(1)
#         if "bad" in filename:
#             output.append(0)
#     return output
#
#
# def file_line_num(fname):
#     with open(fname) as f:
#         for i, l in enumerate(f):
#             pass
#     return i + 1
#
#
# #import data
# x_train = import_skeleton_x_train_data()
# exit()
# y_train = import_skeleton_y_train_data()
#
# x_train = np.array(x_train)
# y_train = np.array(y_train)
#
#
# exit()
#
# #create model
# model = Sequential()
# model.add(LSTM(200, input_shape=(85, 1)))
# model.add(Dense(2, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# print(model.summary())
#
# # Train model
# history = model.fit(x_train, y_train, epochs=10, batch_size=1)
# #train model
# model.fit(x_train, y_train, epochs=4, batch_size=2)
#
#
# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
