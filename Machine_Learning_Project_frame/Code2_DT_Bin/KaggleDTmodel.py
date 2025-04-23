''' This file defines the model classes that will be used. 
    You need to add your code wherever you see "YOUR CODE HERE".
'''

from math import log2
from typing import Protocol

import pandas as pd
import numpy as np

# DON'T CHANGE THE CLASS BELOW! 
# You will implement the train and predict functions in the MajorityBaseline and DecisionTree classes further down.
class Model(Protocol):
    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        ...

    def predict(self, x: pd.DataFrame) -> list:
        ...



class MajorityBaseline(Model):
    def __init__(self):
        pass


    def train(self, x: pd.DataFrame, y: list):
        '''
        Train a baseline model that returns the most common label in the dataset.

        Args:
            x (pd.DataFrame): a dataframe with the features the tree will be trained from
            y (list): a list with the target labels corresponding to each example

        Note:
            - If you prefer not to use pandas, you can convert a dataframe `df` to a 
              list of dictionaries with `df.to_dict(orient='records')`.
        '''

        # YOUR CODE HERE

        #initialize variables
        self.MostCommonLabel = None

        # Find most common label
        self.MostCommonLabel = max(set(y), key = y.count)
        return self.MostCommonLabel

    

    def predict(self, x: pd.DataFrame) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (pd.DataFrame): a dataframe containing the features we want to predict labels for

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE

        # initialize variable
        predict = []

        # find most common label, and create a vector the length of the table. 
        predict = [self.MostCommonLabel] * len(x)
        
        return predict


class DecisionTree(Model):
    def __init__(self, depth_limit: int = None, ig_criterion: str = 'entropy'):
        '''
        Initialize a new DecisionTree

        Args:
            depth_limit (int): the maximum depth of the learned decision tree. Should be ignored if set to None.
            ig_criterion (str): the information gain criterion to use. Should be one of "entropy" or "collision".
        '''
        
        self.depth_limit = depth_limit
        self.ig_criterion = ig_criterion
        self.tree = None


    def train(self, x: pd.DataFrame, y: list):
        '''
        Train a decision tree from a dataset.

        Args:
            x (pd.DataFrame): a dataframe with the features the tree will be trained from
            y (list): a list with the target labels corresponding to each example

        Note:
            - If you prefer not to use pandas, you can convert a dataframe `df` to a 
              list of dictionaries with `df.to_dict(orient='records')`.
            - Ignore self.depth_limit if it's set to None
            - Use the variable self.ig_criterion to decide whether to calulate information gain 
              with entropy or collision entropy
        '''

        # YOUR CODE HERE
       
        # find entropy of the system via entropy

        def findEntropy(y):              
            label_counts = pd.Series(y).value_counts()
            total = len(y)
            entropy = -sum((count / total) * log2(count / total) for count in label_counts)

            return entropy

        def findInformationGainEntropy(x_col, y):
            # convert to pandas series for easier access
            y_series = pd.Series(y).reset_index(drop=True)
            
            # find entropy of the system
            unique_values = x_col.unique()

            # find the total entropy of system
            total_entropy = findEntropy(y)

            # find the entropy of each unique value
            entropy = 0
            for value in unique_values:
                mask = x_col == value
                y_value = y_series[mask.reset_index(drop=True)]
                entropy += len(y_value) / len(y) * findEntropy(y_value)
            InformationGain = total_entropy - entropy
            
            return InformationGain
        
        def findBestSplitEntropy(x,y):
            # set the best information gain to negative infinity and the best split to None
            best_ig = -float('inf')
            best_feature = None

            for col in x.columns:
                ig = findInformationGainEntropy(x[col], y)
                if ig > best_ig:
                    best_ig = ig
                    best_feature = col
        
            return best_feature
        
        def buildTreeEntropy(x, y, depth=0):
            # Base case when all labels are the same
            if len(set(y)) == 1:
                print(f"Leaf node at depth {depth} with label {y[0]}")
                return y[0]

            # If we reach depth limit, create a leaf node
            if self.depth_limit is not None and depth >= self.depth_limit:
                most_common = max(set(y), key=y.count)
                print(f"Depth limit reached, returning majority class {most_common}")
                return most_common

            # Otherwise, find the best feature and recursively build the tree
            best_feature = findBestSplitEntropy(x, y)
            tree = {best_feature: {}}

            # Split data by the best feature and recursively build the tree
            for value in x[best_feature].unique():
                subset_x = x[x[best_feature] == value].drop(columns=[best_feature])
                subset_y = [y[i] for i in range(len(x)) if x.iloc[i][best_feature] == value]
                tree[best_feature][value] = buildTreeEntropy(subset_x, subset_y, depth + 1)

            return tree

        # find entropy of the system via collision entropy

        def findCollisionEntropy(y):
            label_counts = pd.Series(y).value_counts()
            total = len(y)

            probabilities = label_counts / total

            # collision entropy
            prob_squared = np.sum(probabilities ** 2)
            collision_entropy = -np.log2(prob_squared)
            return collision_entropy
        
        def findInformationGainCollsionEntropy(x_col, y):
            # convert to pandas series for easier access
            y_series = pd.Series(y).reset_index(drop=True)
            
            # find entropy of the system
            unique_values = x_col.unique()

            # find the total entropy of system
            total_entropy = findEntropy(y)

            # find the entropy of each unique value
            entropy = 0
            for value in unique_values:
                mask = x_col == value
                y_value = y_series[mask.reset_index(drop=True)]
                entropy += len(y_value) / len(y) * findCollisionEntropy(y_value)
            InformationGain = total_entropy - entropy
            
            return InformationGain
        
        def findBestSplitCollsionEntropy(x,y):
            # set the best information gain to negative infinity and the best split to None
            best_ig = -float('inf')
            best_feature = None

            for col in x.columns:
                ig = findInformationGainCollsionEntropy(x[col], y)
                if ig > best_ig:
                    best_ig = ig
                    best_feature = col
        
            return best_feature
        
        def buildTreeCollsionEntropy(x, y, depth=0):
            # Base case when all labels are the same
            if len(set(y)) == 1:
                return y[0]

            # If we reach depth limit, create a leaf node
            if self.depth_limit is not None and depth >= self.depth_limit:
                return max(set(y), key=y.count)

            # Otherwise, find the best feature and recursively build the tree
            best_feature = findBestSplitCollsionEntropy(x, y)
            tree = {best_feature: {}}

            # Split data by the best feature and recursively build the tree
            for value in x[best_feature].unique():
                subset_x = x[x[best_feature] == value].drop(columns=[best_feature])
                subset_y = [y[i] for i in range(len(x)) if x.iloc[i][best_feature] == value]
                tree[best_feature][value] = buildTreeCollsionEntropy(subset_x, subset_y, depth + 1)

            return tree




        # if the depth limit is None, set it to infinity
        if self.depth_limit == None:
            self.depth_limit = float('inf')
        
        # if the critierion is entropy use entropy tree

        if self.ig_criterion == 'entropy':
            self.tree = buildTreeEntropy(x, y)
            print(f"Tree structure: {self.tree}")  # Debug tree output
        # if the criterion is collision use collision tree    
        elif self.ig_criterion == 'collision':
            self.tree = buildTreeCollsionEntropy(x, y)
            pass
        else:
            raise ValueError('Invalid criterion')
            
      

    def predict(self, x: pd.DataFrame) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (pd.DataFrame): a dataframe containing the features we want to predict labels for

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''
        predictions = []

        # Helper function to make predictions for a single row
        def predict_single(row, tree):
            if isinstance(tree, dict):
                # Get the feature that the tree is split on
                feature = list(tree.keys())[0]
                # Find the value for this feature in the row
                feature_value = row[feature]
                
                # Recursively call the function on the appropriate branch based on feature value
                if feature_value in tree[feature]:
                    return predict_single(row, tree[feature][feature_value])
                else:
                    # If a feature value doesn't exist in the tree, predict the most common label in the branch
                    # Instead of iterating over the branch, just return the most frequent label
                    # Get all leaf labels in the tree and return the most frequent one
                    leaf_labels = [predict_single(row, subtree) for subtree in tree[feature].values()]
                    return max(set(leaf_labels), key=leaf_labels.count)
            else:
                # If we are at a leaf node, return the label
                return tree

        # Iterate over each row in the dataframe and predict the label
        for _, row in x.iterrows():
            predictions.append(predict_single(row, self.tree))

        return predictions
