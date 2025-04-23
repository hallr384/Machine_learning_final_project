''' This file contains the functions for training and evaluating a model.
    You need to add your code wherever you see "YOUR CODE HERE".
'''

import argparse
import numpy as np
import pandas as pd

from KaggleDTdata import load_data
from KaggleDTmodel import DecisionTree, MajorityBaseline, Model
from KaggleConvertToCategory import convert

convert() 

def train(model: Model, x: pd.DataFrame, y: list):
    '''
    Learn a model from training data.

    Args:
        model (Model): an instantiated MajorityBaseline or DecisionTree model
        x (pd.DataFrame): a dataframe with the features the tree will be trained from
        y (list): a list with the target labels corresponding to each example
    '''
    
    # YOUR CODE HERE
    model.train(x, y)
    print("Training complete.")


def evaluate(model: Model, x: pd.DataFrame, y: list) -> float:
    '''
    Evaluate a trained model against a dataset

    Args:
        model (Model): an instance of a MajorityBaseline model or a DecisionTree model
        x (pd.DataFrame): a dataframe with the features the tree will be trained from
        y (list): a list with the target labels corresponding to each example

    Returns:
        float: the accuracy of the decision tree's predictions on x, when compared to y
    '''
    
    # YOUR CODE HERE
    predictions = model.predict(x)
    print(f'Predictions pre np.array: {predictions}')

    predictions = np.array(predictions)

    print(f'Predictions: {predictions}')
    print(f'Predictions type: {type(predictions)}')

    accuracy = calculate_accuracy(y, predictions)
    return accuracy,predictions


def calculate_accuracy(labels: list, predictions: list) -> float:
    '''
    Calculate the accuracy between ground-truth labels and candidate predictions.
    Should be a float between 0 and 1.

    Args:
        labels (list): the ground-truth labels from the data
        predictions (list): the predicted labels from the model

    Returns:
        float: the accuracy of the predictions, when compared to the ground-truth labels
    '''

    # YOUR CODE HERE
    
    #compare labels and predictions, then calculate accuracy
    correct = 0
    print(f'label shape {len(labels)}')
    print(f'prediction shape {len(predictions)}')

    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            correct += 1
    accuracy = correct / len(labels)
    return accuracy



# DON'T EDIT ANY OF THE CODE BELOW
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a model')
    parser.add_argument('--model_type', '-m', type=str, choices=['majority_baseline', 'decision_tree'], 
        help='Which model type to train')
    parser.add_argument('--depth_limit', '-d', type=int, default=None, 
        help='The maximum depth of a DecisionTree. Ignored if model_type is not "decision_tree".')
    parser.add_argument('--ig_criterion', '-i', type=str, choices=['entropy', 'collision'], default='entropy',
        help='Which information gain variant to use. Ignored if model_type is not "decision_tree".')
    args = parser.parse_args()


    # load data
    data_dict = load_data()

    train_df = data_dict['train']
    train_x = train_df.drop('label', axis=1)
    train_y = train_df['label'].tolist()

    test_df = data_dict['test']
    test_x = test_df.drop('label', axis=1)
    test_y = test_df['label'].tolist()

    anon_df = data_dict['anon']
    anon_x = anon_df.drop('label', axis=1)
    anon_y = anon_df['label'].tolist()

    #print(f'Training Data Types: {train_x.dtypes}')
    #print(f'Prediction Data Types: {test_x.dtypes}')

    # initialize the model
    if args.model_type == 'majority_baseline':
        model = MajorityBaseline()
    elif args.model_type == 'decision_tree':
        model = DecisionTree(depth_limit=args.depth_limit, ig_criterion=args.ig_criterion)
    else:
        raise ValueError(
            '--model_type must be one of "majority_baseline" or "decision_tree". ' +
            f'Received "{args.model_type}". ' +
            '\nRun `python train.py --help` for additional guidance.')



    
    # train the model
    print(f"Training data shape: {train_x.shape}, Labels: {len(train_y)}")
    train(model=model, x=train_x, y=train_y)


    # save train and test to csv
    train_x.to_csv('train_data.csv', index=False)
    test_x.to_csv('test_data.csv', index=False)
    anon_x.to_csv('anon_data.csv', index=False)

    # evaluate model on train and test data
    train_accuracy,train_predictions = evaluate(model=model, x=train_x, y=train_y)
    print(f'train accuracy: {train_accuracy:.3f}')

    print(f"testing data shape: {test_x.shape}, Labels: {len(test_y)}")

    print(f'Running Test')
    test_accuracy,test_predictions = evaluate(model=model, x=test_x, y=test_y)
    print(f'test accuracy: {test_accuracy:.3f}')

    # evaluate model on eval.anon data
    anon_accuracy,anon_predictions = evaluate(model=model, x=anon_x, y=anon_y)
    print(f'anon accuracy: {anon_accuracy:.3f}')

    # save y_pred for anon data to csv

    # get eval.id from eval.id and make csv with Id in one collumn and anon predictions the second collumn
    eval_id = pd.read_csv('project_data/data/eval_id')
    Results = pd.DataFrame()
    Results['example_id'] = eval_id
    Results['label'] = anon_predictions
    Results.to_csv('./KaggleResults/DT_predictions.csv', index=False)
    print('Predictions saved to DT_predictions.csv')


    
    print(f'train accuracy: {train_accuracy:.3f}')
    print(f'test accuracy: {test_accuracy:.3f}')
    print(f'anon accuracy: {anon_accuracy:.3f}')

    

    
        
