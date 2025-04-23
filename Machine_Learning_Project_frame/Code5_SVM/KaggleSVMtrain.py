
''' This file contains the code for training and evaluating a model.
    You don't need to change this file.
'''

import argparse

from KaggleSVMdata import load_data
from KaggleSVMevaluate import accuracy
from KaggleSVMmodel import LogisticRegression, MajorityBaseline, Model, SupportVectorMachine, MODEL_OPTIONS
import numpy as np
import pandas as pd
def init_model(args: object, num_features: int) -> Model:
    '''
    Initialize the appropriate model from command-line arguments.

    Args:
        args (object): the argparse Namespace mapping arguments to their values.
        num_features (int): the number of features (i.e. dimensions) the model will have

    Returns:
        Model: a Model object initialized with the hyperparameters in args.
    '''

    if args.model == 'majority_baseline':
        model = MajorityBaseline()
    
    elif args.model == 'svm':
        model = SupportVectorMachine(
            num_features=num_features, 
            lr0=args.lr0, 
            C=args.reg_tradeoff)

    elif args.model == 'logistic_regression':
        model = LogisticRegression(
            num_features=num_features, 
            lr0=args.lr0, 
            sigma2=args.reg_tradeoff)
    
    return model


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
    print(' made it to evaluate')
    # print the shape of x and y
    print(f'x shape: {x.shape}')
    print(f'y shape: {len(y)}')
    # get predictions from model row by row
    
    # intialize
    predictions = []
    # iterate through each row
    for i in range(len(x)):
        # get the row
        row = x[i]
        # convert to numpy array
        row = np.array(row)
        # reshape to 1D array
        row = row.reshape(1, -1)
        # get the prediction
        prediction = model.predict(row)

        # convert to int
        prediction = int(prediction[0])
        # append to predictions
        predictions.append(prediction)
        
    
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
    parser.add_argument('--model', '-m', type=str, choices=MODEL_OPTIONS, 
        help=f'Which model to run. Must be one of {MODEL_OPTIONS}.')
    parser.add_argument('--lr0', type=float, default=0.1, 
        help='The initial learning rate hyperparameter gamma_0. Defaults to 0.1.')
    parser.add_argument('--reg_tradeoff', type=float, default=1, 
        help='The regularization tradeoff hyperparameter for SVM and Logistic Regression. Defaults to 1.')
    parser.add_argument('--epochs', '-e', type=int, default=20,
        help='How many epochs to train for. Defaults to 20.')
    args = parser.parse_args()

    # load data
    data_dict = load_data()

    train_df = data_dict['train']
    train_x = train_df.drop('label', axis=1)
    train_y = train_df['label'].tolist()

    # convert to np.array
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    print('after loading train')
    print(f'test x shape: {train_x.shape}')
    print(f'test y shape: {train_y.shape}')

    test_df = data_dict['test']
    test_x = test_df.drop('label', axis=1)
    test_y = test_df['label'].tolist()
    # convert to np.array
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    print('after loading test')
    print(f'test x shape: {test_x.shape}')
    print(f'test y shape: {test_y.shape}')


    anon_df = data_dict['anon']
    anon_x = anon_df.drop('label', axis=1)
    anon_y = anon_df['label'].tolist()
    # convert to np.array
    anon_x = np.array(anon_x)
    anon_y = np.array(anon_y)

    # initialize model
    print('initialize model')
    model = init_model(args=args, num_features=train_x.shape[1])
    # train the model
    if args.model == 'majority_baseline':
        print(f'train model')
        model.train(x=train_x, y=train_y)
    else:
        print(f'train model for {args.epochs} epochs')
        model.train(x=train_x, y=train_y, epochs=args.epochs)

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

    print(f'anon data shape: {anon_x.shape}, Labels: {len(anon_y)}')
    print('anon prediction size: ', len(anon_predictions))
    # save y_pred for anon data to csv

    # get eval.id from eval.id and make csv with Id in one collumn and anon predictions the second collumn
    eval_id = pd.read_csv('project_data/data/eval_id')
    Results = pd.DataFrame()
    Results['example_id'] = eval_id
    Results['label'] = anon_predictions
    Results.to_csv('./KaggleResults/SVM_predictions.csv', index=False)
    print('Predictions saved to SVM_predictions.csv')


    
    print(f'train accuracy: {train_accuracy:.3f}')
    print(f'test accuracy: {test_accuracy:.3f}')
    print(f'anon accuracy: {anon_accuracy:.3f}')