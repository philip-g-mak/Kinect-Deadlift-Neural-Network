#imports
import scipy.io
import numpy as np
import pandas as pd
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from pathlib import Path
from matplotlib import pyplot as plt
from keras.utils import to_categorical



#skeleton data Process
def import_skeleton_data(action, subject, trial):
    filename = f'data/Skeleton/a{action}_s{subject}_t{trial}_skeleton.mat'
    if Path(filename).is_file():
        mat = scipy.io.loadmat(filename)
        return mat['d_skel']
    else:
        return None

def transform_skeleton_data(action, subject, trial):
    matrices = []
    data = import_skeleton_data(action, subject, trial)
    if data is None: return None
    for frame in range(data.shape[2]):
        skelecton_joints = [i + 1 for i in range(20)]
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

def export_inertial_data_to_csv(action, subject, trial):
    df = transform_skeleton_data_to_df(action, subject, trial)
    if df is None: return None
    filename = f'a{action}_s{subject}_t{trial}_skeleton.csv'
    df.to_csv(filename, index=False)

#Classification using Intertial Data
# Original inertial data has dimension (Number of sample) x 6
# Swap the axes so the new dimension is 6 x (Number of sample)
# Apply padding to each entry, the new dimension is 6 x 326
# Subjects 1, 2, 3, 5, 6, 7 go into training data (75%)
# Subjects 4, 8 go into test data (25%)

X_train = []
Y_train = []
X_test = []
Y_test = []

#     activites for training:
#     1. Deadlift Conventional Correct
#     11. Deadlift Sumo Corect
#
activities = [1, 11]

for index, action in enumerate(activities):
    for subject in range(1, 9):
        for trial in range(1, 5):
            data = import_skeleton_data(action, subject, trial)
            if data is None: continue
            data = np.swapaxes(data, 0, 1)
            data = sequence.pad_sequences(data, maxlen=326)
            if subject in [1, 2 ,3, 5, 6, 7] :
                X_train.append(data)
                Y_train.append(index)
            else:
                X_test.append(data)
                Y_test.append(index)

a = array(X_train)
print(a.shape)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

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
model.add(LSTM(200, input_shape=(326, 6)))
model.add(Dense(len(activities), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
print(model.summary())

# Train model
history = model.fit(X_train, Y_train, epochs=10, batch_size=1)

# Evaluate model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

import seaborn as sns
sns.set(style="darkgrid")
plt.plot(history.history['acc'])
plt.title('RNN LSTM - Activity Classification')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
