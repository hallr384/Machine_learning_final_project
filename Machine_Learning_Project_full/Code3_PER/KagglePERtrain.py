''' This file contains the code for training and evaluating a model.
    You don't need to change this file.
'''

import argparse
import pandas as pd
import numpy as np


from KagglePERdata import load_data
from KagglePERevaluate import accuracy
from KagglePERmodel import init_perceptron, MajorityBaseline, MODEL_OPTIONS,Model



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
    parser.add_argument('--model', '-m', type=str, default='simple', choices=MODEL_OPTIONS, 
        help=f'Which model to run. Must be one of {MODEL_OPTIONS}.')
    parser.add_argument('--lr', type=float, default=1, 
        help='The learning rate hyperparameter eta (same as the initial learning rate). Defaults to 1.')
    parser.add_argument('--mu', type=float, default=0, 
        help='The margin hyperparameter mu. Defaults to 0.')
    parser.add_argument('--epochs', '-e', type=int, default=10,
        help='How many epochs to train for. Defaults to 10.')
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



    # debug by showing y values for train, test, and anon
    #print(f'  train y: {train_y}')
    #print(f'  test y: {test_y}')
    #print(f'  anon y: {anon_y}')
    # load model using helper function init_perceptron() from model.py
    print(f'initialize model')
    if args.model == 'majority_baseline':
        model = MajorityBaseline()

        # train the model
        print(f'train MajorityBaseline')
        model.train(x=train_x, y=train_y)
    
    else:
        model = init_perceptron(
            variant=args.model, 
            num_features=train_x.shape[1], 
            lr=args.lr, 
            mu=args.mu)
        print(f'  model type: {type(model).__name__}\n  hyperparameters: {model.get_hyperparams()}')

        # train the model
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
    Results.to_csv('./KaggleResults/PER_predictions.csv', index=False)
    print('Predictions saved to PER_predictions.csv')


    
    print(f'train accuracy: {train_accuracy:.3f}')
    print(f'test accuracy: {test_accuracy:.3f}')
    print(f'anon accuracy: {anon_accuracy:.3f}')







