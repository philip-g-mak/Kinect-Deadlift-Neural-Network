# Kinect-Deadlift-Neural-Network

2018 Senior Design Project - Rutgers University
A neural network utilizing Microsoft Kinnect's motion tracking abilities to determine correct and incorrect
lifting form.  This project can be applied to any movement or motion given a large and varied enough
dataset.

FOR TRAINING
    Store Kinect Data with naming convention 
    {b for bad form | g for good forma}_{trial number}
    in /Skeleton_Data/raw_data

    run kinect_to_csv.py to convert kinnect data to a readable csv file
    $ python kinect_to_csv.py

    run csv_to_mat.py to convert csv data to numpy array
    $ python csv_to_numpy.py
    or whatever directory your csv data is located
    
    then once all the data is formatted correctly
    $ python Deadlift_Analyzer.py
    
    the model will be saved as DeadList_Model.h5

FOR EVALUATING
Store Kinect Data in Evaluate_Input/raw_data with any name (somewhat intelligent pls)
then run
    $ python Evaluate_Input.py 
