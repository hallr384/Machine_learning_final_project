import pandas as pd
import numpy as np
import os
from glob import glob


def load_data():
    '''
    Loads and prepares the dataset for training and testing.
    This function returns a dictionary containing 'train' and 'test' DataFrames.
    '''
    # Define the paths to the training and testing datasets
    train_data_path = './project_data/data/train_cat.csv'
    test_data_path = './project_data/data/test_cat.csv'
    anon_data_path = './project_data/data/evalanon_cat.csv'

    # Check if files exist at the specified paths
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Data files not found at {train_data_path} or {test_data_path}")

    # Load the data from the CSV files
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    anon_df = pd.read_csv(anon_data_path)

    # create folds
    

    ## load cross validation datasets
    #cv_folds = []
    #for cv_fold_path in glob('data/cv/*'):
    #    fold = pd.read_csv(cv_fold_path)
    #    cv_folds.append(fold)

    # Process categories to handle out-of-bound values
    #categories_dict = {}
    #for column in train_df.columns:
    #    if train_df[column].dtype == 'object':  # Handling categorical columns
    #        categories = sorted(train_df[column].unique())
    #        categories_dict[column] = categories
    #        train_df[column] = train_df[column].apply(lambda x: map_to_nearest_category(x, categories))

    # Map the test data to the same categories
    #for column in test_df.columns:
    #    if column in categories_dict:
    #        test_df[column] = test_df[column].apply(lambda x: map_to_nearest_category(x, categories_dict[column]))

    # Map the anon data to the same categories
    #for column in anon_df.columns:
    #    if column in categories_dict:
    #        anon_df[column] = anon_df[column].apply(lambda x: map_to_nearest_category(x, categories_dict[column]))
    

    ## Debugging: Print categories for both train and test data
    #print("Categories from train data:")
    #for column, categories in categories_dict.items():
    #    print(f"{column}: {categories}")
    
    
    #print("\nSample data from train and test (mapped to categories):")
    #print("Train data sample:")
    #print(train_df.head())
    #print("\nTest data sample:")
    #print(test_df.head())

    return {'train': train_df, 'test': test_df , 'anon': anon_df}


#def map_to_nearest_category(value, categories):
    '''
    Adjusts a value to the nearest category if it is out of the expected range.
    
    Args:
        value: The value to be mapped to the nearest category.
        categories: A list of possible categories, sorted in ascending order.
    
    Returns:
        The closest category to the input value.
    '''
#    if value not in categories:
#        # If the value is outside the range of the categories, map it to the nearest one
#        value = min(categories, key=lambda x: abs(x - value))
#    return value
