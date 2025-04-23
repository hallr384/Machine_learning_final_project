''' This file provides utility functions for loading data that you may find useful.
    You don't need to change this file.
'''

from glob import glob

import pandas as pd
import numpy as np
import os

def load_data() -> dict:
    '''
    Loads all the data required for this assignment.

    Returns:
        dict: a dictionary containing the train, test, and cv data as (x, y) tuples of np.ndarray matrices
    '''

    # Define the paths to the training and testing datasets
    train_data_path = './project_data/data/train.csv'
    test_data_path = './project_data/data/test.csv'
    anon_data_path = './project_data/data/evalanon.csv'

    # Check if files exist at the specified paths
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Data files not found at {train_data_path} or {test_data_path}")

    # Load the data from the CSV files
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    anon_df = pd.read_csv(anon_data_path)

    # load train dataset
    train = train_df
    # save to float
    train = train.astype(np.float32)

    # load validation dataset
    val = test_df
    # save to float
    val = val.astype(np.float32)

    # load test dataset
    test = anon_df
    # save to float
    test = test.astype(np.float32)

    # load cross validation datasets
    #cv_folds = []
    #for cv_fold_path in glob('data/cv/*'):
    #    fold = pd.read_csv(cv_fold_path)
    #    cv_folds.append(fold)

    print('Data loaded successfully')
    # Print the shapes of the loaded datasets
    print(f'Train shape: {train.shape}')
    print(f'Test shape: {val.shape}')
    print(f'Anon shape: {test.shape}')

    return {
        'train': train,
        'test': val,
        'anon': test}#,
        #'cv_folds': cv_folds}
