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


def transform_kinect_files_to_csv():
    raw_data_directory  = f'Evaluate_Input/raw_data/'
    process_data_directory =  f'Evaluate_Input/csv_data/'
    for filename in os.listdir(raw_data_directory):
        if "DS_Store" not in filename:
            if "b" in filename:
                processed_filename = "test_input"
            else:
                processed_filename = "test_input"




                f= open(process_data_directory + "/"+ processed_filename + ".csv","w+")
                f.write("action,subject,trial,frame,skeleton_joint,x,y,z\n")
                with open('{}/{}'.format(raw_data_directory, filename)) as current_file_f:
                    current_file = current_file_f.readlines()
                    frame_count = 0
                    for frame in current_file[:-1]:
                        joint_count = 1
                        for xyz_coords_str in frame.replace(";\n","").split(';'):
                            xyz_coords = xyz_coords_str.split(',')
                            try:
                                x = float(xyz_coords[0]);
                                y = float(xyz_coords[1]);
                                z = float(xyz_coords[2]);


                                output = '0.0, ' + str(0) + ', ' + str(0) + ', ' + str(frame_count) + ', ' + str(joint_count) + ',' + str(x) + ', ' + str(y) +', ' + str(z) + '\n'
                                f.write(output)
                            except:
                                continue
                            #     print(filename)
                            #     print(xyz_coords)
                            #     exit()
                            joint_count = joint_count + 1;
                        frame_count = frame_count + 1;

        else:
            continue

def transform_csv_to_numpy():
    csv_data_directory  = f'Evaluate_Input/csv_data/'
    process_data_directory =  f'Evaluate_Input/matrix_numpy_data/'

    for filename in os.listdir(csv_data_directory):
        if "DS_Store" not in filename:
            output_data = []

            processed_filename = filename[:-4]

            f= open('{}{}'.format(process_data_directory, processed_filename),"w+")
            datafile = open('{}{}'.format(csv_data_directory, filename), 'r')

            data = genfromtxt(datafile, delimiter=',')
            for i in range(1, 17):
                joint_data = []

                x_data = []
                y_data = []
                z_data = []
                for j in range(1, len(data)):
                    if data[j][4] == i and i < 17:
                        x_data.append(data[j][5])
                        y_data.append(data[j][6])
                        z_data.append(data[j][7])

                joint_data.append(x_data)
                joint_data.append(y_data)
                joint_data.append(z_data)
                output_data.append(joint_data)
            output_data = np.asarray(output_data)
            np.save('{}{}'.format(process_data_directory, processed_filename), output_data)


def main():
    transform_kinect_files_to_csv()
    transform_csv_to_numpy()
    model = load_model('my_model.h5')

    numpy_data_dir = f'Evaluate_Input/matrix_numpy_data/'
    for filename in os.listdir(numpy_data_dir):
        if "DS_Store" not in filename:
            data = np.load(filename)
            data = np.swapaxes(data, 0, 1)
            data = sequence.pad_sequences(data, maxlen=326)
            data = np.array(data)
            data = np.swapaxes(data, 1, 2)


            output = model.predict(self, data, batch_size= None, verbose=0, steps=None)
            print(output)



if __name__ == "__main__":
    main()
