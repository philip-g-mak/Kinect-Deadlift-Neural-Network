import scipy.io
import numpy as np
import pandas as pd
from numpy import array, genfromtxt
import csv
import os

np.random.seed(123)  # for reproducibility

NUMBER_OF_SUBJECTS = 10
NUMBER_OF_TRIALS = 5 #per subject
def transform_kinect_files_to_matrix():
    csv_data_directory  = f'Skeleton_Data/csv_data/'
    process_data_directory =  f'Skeleton_Data/matrix_numpy_data/'

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
transform_kinect_files_to_matrix()
