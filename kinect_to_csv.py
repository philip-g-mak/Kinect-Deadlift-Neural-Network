import scipy.io
import numpy as np
import pandas as pd
from numpy import array
import csv
import os

np.random.seed(123)  # for reproducibility

NUMBER_OF_SUBJECTS = 10
NUMBER_OF_TRIALS = 5 #per subject
def transform_kinect_files_to_csv():
    raw_data_directory  = f'Skeleton_Data/raw_data/'
    process_data_directory =  f'Skeleton_Data/csv_data/'
    for filename in os.listdir(raw_data_directory):
        if "DS_Store" not in filename:
            if "b" in filename:
                processed_filename = filename.replace("b_","a11_s")
            else:
                processed_filename = filename.replace("g_","a1_s")

            subject_string  = processed_filename[-2:]
            subject_string = subject_string. replace("s","")
            subject_string = subject_string.replace("_","")
            subject_int = int(subject_string) % NUMBER_OF_SUBJECTS

            processed_filename =  processed_filename.replace(subject_string, str(subject_int))
            for trial_count in range(NUMBER_OF_TRIALS):
                output_filename = processed_filename + "_t" + str(trial_count) + "_skeleton"

                f= open(process_data_directory + "/"+ output_filename + ".csv","w+")
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

                                if "a11" in output_filename:
                                    output = '11.0, ' + str(subject_int) + ', ' + str(trial_count) + ', ' + str(frame_count) + ', ' + str(joint_count) + ',' + str(x) + ', ' + str(y) +', ' + str(z) + '\n'
                                    f.write(output)
                                else:
                                    output = '1.0, ' + str(subject_int) + ', ' + str(trial_count) + ', ' + str(frame_count) + ', ' + str(joint_count) + ',' + str(x) + ', ' + str(y) +', ' + str(z) + '\n'
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



transform_kinect_files_to_csv()
