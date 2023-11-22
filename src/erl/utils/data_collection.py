import os 
import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd
import pickle 

odomdim = 14 
erldim = 7

file_name = 'data/jackal_dataset.pkl'

# Specify the columns you want to select from each file
odom_columns = ['field.header.stamp', 'field.pose.pose.position.x', 'field.pose.pose.position.y', 'field.pose.pose.position.z', 
                'field.pose.pose.orientation.x', 'field.pose.pose.orientation.y', 'field.pose.pose.orientation.z', 'field.pose.pose.orientation.w',
                'field.twist.twist.linear.x', 'field.twist.twist.linear.y', 'field.twist.twist.linear.z',
                'field.twist.twist.angular.x', 'field.twist.twist.angular.y', 'field.twist.twist.angular.z']
erl_columns = ['field.left_desired_velocity', 'field.right_desired_velocity', 'field.left_measured_velocity', 'field.right_measured_velocity', 'field.left_desired_torque', 'field.right_desired_torque', 'field.odom.header.stamp']


def get_dataset(file_one, file_two, filter='no'):
    if os.path.exists(file_name):
        # Load the result from the pickle file
        with open(file_name, 'rb') as file:
            result_array = pickle.load(file)
        print('Loaded result from', file_name)
    else:
        # Load CSV files
        odom = pd.read_csv(file_one, usecols=odom_columns, dtype={0: float})
        erl = pd.read_csv(file_two, usecols=erl_columns, dtype={0: float})

        # Assuming the timestamp is in the first column for both files
        odom_timestamps = odom.iloc[:, 0].astype('int64')
        erl_timestamps = erl.iloc[:, -1].astype('int64')

        # Initialize results
        result = []

        # Pointers for odom and erl
        i_odom = 0
        i_erl = 0

        # Iterate through the odom timestamps
        while i_odom < len(odom_timestamps) and i_erl < len(erl_timestamps)-1:
            if odom_timestamps[i_odom] < erl_timestamps[i_erl]:
                i_odom += 1
            elif odom_timestamps[i_odom] >= erl_timestamps[i_erl] and odom_timestamps[i_odom] < erl_timestamps[i_erl+1]:
                # Get the samples from erl and odom
                sample_erl = erl.iloc[i_erl, :]
                sample_odom = odom.iloc[i_odom, :]

                # Combine erl and odom samples
                combined_sample = np.hstack([sample_odom, sample_erl, i_erl])
                result.append(combined_sample)
                i_odom += 1

            else:
                i_erl += 1
        # Convert the result to a NumPy array
        result_array = np.array(result)
        # Save the result as a pickle file
        with open(file_name, 'wb') as file:
            pickle.dump(result_array, file)
        print('Saved result to result.pkl')
    # Process dataset 
    if (filter == 'yes'):
        result_array = filter_velocity(result_array, alpha=0.95)
    result_array = add_prediction_steps(result_array) # add one more dimension 
    
    return result_array

def add_prediction_steps(dataset):
    '''
    new_dataset: (time, pos, quat, v, w, time, u, pred_steps, dt)  
    '''
    samples = dataset.shape[0]
    dim = dataset.shape[1]
    new_dataset =  np.zeros((samples, dim + 1)) #
    cur_val = 0.0
    pred_steps = 0

    print(f'dim = {dim}')

    new_dataset[:, 0:dim-1] = dataset[:, 0:dim-1]

    for i in range(samples-1):
        cur_val = dataset[i, -1]
        next_val = dataset[i+1, -1]
        if (cur_val == next_val):
            pred_steps += 1
        else:
            new_dataset[i-pred_steps, -2] = pred_steps + 1 
            # add dt 
            t_start = dataset[i-pred_steps, 0]
            for k in range(pred_steps):
                new_dataset[i-pred_steps+k+1, -1] = (dataset[i-pred_steps+k+1, 0] - t_start) * 1e-9
            pred_steps = 0
    return new_dataset

def get_specific_prediction_steps(dataset, prediction_steps=5):
    '''
    '''
    pred_steps = prediction_steps
    samples = dataset.shape[0]
    dim = dataset.shape[1] - 1
    total = np.sum(np.count_nonzero(dataset[:, -2] == pred_steps))
    new_dataset = np.zeros((pred_steps, total, dim))
    
    i = 0 
    idx = 0
    while idx < samples and i < total:
        val = dataset[idx, -2]
        if (val == pred_steps):
            for j in range(pred_steps):
                new_dataset[j, i, :] = np.concatenate((dataset[idx, 0:dim-1], np.expand_dims(dataset[idx, dim], axis=0)))
                idx += 1
            i += 1
        else:
            idx += 1
    new_dataset = process_dataset(new_dataset)
    return new_dataset

def process_dataset(dataset):
    pred_steps = dataset.shape[0]
    samples = dataset.shape[1]
    dim = 21 # 18 + 2 controls + 1 dt   
    new_dataset = np.zeros((pred_steps, samples, dim))
    for i in range(samples):
        for j in range(pred_steps):
            _, x, quat, v, w, _, u, _, dt = np.split(dataset[j, i, :], [1, 4, 8, 11, 14, 15, 19, 21]) # xdim, qdim, vdim, wdim, udim
            input = np.array([u[0] - u[2], u[1] - u[3]])
            rotation = Rotation.from_quat(quat)
            rotation_matrix = rotation.as_matrix()
            new_dataset[j, i, :] = np.concatenate((x, rotation_matrix.flatten(), v, w, input, dt))
    return new_dataset


def filter_velocity(dataset, alpha=0.8):
    alpha = alpha
    N = dataset.shape[0]
    dim = dataset.shape[1]
    new_dataset = np.zeros((N, dim))
    # Get linear velocity and angular velocity 
    vidx = 8 
    wdix = 11
    v = dataset[:, vidx:vidx+3]
    w = dataset[:, wdix:wdix+3]
    
    # linear velocity 
    cur_linvel = v[0]
    cur_angvel = w[0]
    prev_linvel = cur_linvel
    prev_angvel = cur_angvel
    new_dataset = dataset
    new_dataset[0, vidx:vidx+3] = cur_linvel
    new_dataset[0, wdix:wdix+3] = cur_angvel
    for k in range(N-1):
        cur_linvel = (alpha * prev_linvel) + (1 - alpha) * v[k+1]
        cur_angvel = (alpha * prev_angvel) + (1 - alpha) * w[k+1]

        new_dataset[k+1, vidx:vidx+3] = cur_linvel
        new_dataset[k+1, wdix:wdix+3] = cur_angvel

        prev_linvel = cur_linvel
        prev_angvel = cur_angvel
    
    return new_dataset