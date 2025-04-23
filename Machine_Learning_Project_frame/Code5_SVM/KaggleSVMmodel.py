
''' This file defines the model classes that will be used. 
    You need to add your code wherever you see "YOUR CODE HERE".
'''

from typing import Protocol

import numpy as np

from utils import clip, shuffle_data




# set the numpy random seed so our randomness is reproducible
np.random.seed(1)

MODEL_OPTIONS = ['majority_baseline', 'svm', 'logistic_regression']


# DON'T CHANGE THE CLASS BELOW! 
# You will implement the train and predict functions in the classes further down.
class Model(Protocol):
    def __init__(**hyperparam_kwargs):
        ...

    def get_hyperparams(self) -> dict:
        ...

    def loss(self, ) -> float:
        ...
        
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        ...

    def predict(self, x: np.ndarray) -> list:
        ...



class MajorityBaseline(Model):
    def __init__(self):
        
        # YOUR CODE HERE, REMOVE THE LINE BELOW
        self.MostCommonLabel = None
        
        


    def get_hyperparams(self) -> dict:
        return {}
    

    def loss(self, x_i: np.ndarray, y_i: int) -> float:
        return None
    

    def train(self, x: np.ndarray, y: np.ndarray):
        '''
        Train a baseline model that returns the most common label in the dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example

        Hints:
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
        '''
    
        # YOUR CODE HERE
        
        # determine the number of each unique label
        unique_labels, counts = np.unique(y, return_counts=True)

        # find the most common label
        self.MostCommonLabel = unique_labels[np.argmax(counts)]
        return self.MostCommonLabel



    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW

        # initialize variable
        predict = []

        # find most common label, and create a vector the length of the table. 
        predict = [self.MostCommonLabel] * len(x)

        return predict
    


class SupportVectorMachine(Model):
    def __init__(self, num_features: int, lr0: float, C: float):
        '''
        Initialize a new SupportVectorMachine model

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            lr0 (float): the initial learning rate (gamma_0)
            C (float): the regularization/loss tradeoff hyperparameter
        '''

        self.lr0 = lr0
        self.C = C

        # YOUR CODE HERE

        # initialize the weights to a random number which is small to start
        self.w = np.random.rand(num_features) *0.01
        self.b =0

        # initialize t, the number of iterations
        self.t = 0

        # plot or not
        #self.plotting = True
        self.plotting = False        


    
    def get_hyperparams(self) -> dict:
        return {'lr0': self.lr0, 'C': self.C}
    

    def loss(self, x_i: np.ndarray, y_i: int) -> float:
        '''
        Calculate the SVM loss on a single example.

        Args:
            x_i (np.ndarray): a 1-D np.ndarray (num_features) with the features for a single example
            y_i (int): the label for the example, either 0 or 1.

        Returns:
            float: the loss for the example using the current weights

        Hints:
            - Don't forget to convert the {0, 1} label to {-1, 1}.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        # initialize variable
        total_loss = 0
        
        # convert the label to {-1, 1}
        if y_i == 0:
            y_i = -1
        else:
            y_i = 1
        
        # find margin
        marg = y_i * (np.dot(self.w, x_i) + self.b)

        # calculate the hinge loss
        hinge_loss = max(0, 1 - marg)

        # calculate the regularization loss
        reg_loss = 0.5 * np.dot(self.w, self.w)

        # calculate the total loss
        total_loss =  reg_loss + self.C * hinge_loss 

        


        return total_loss
    
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Shuffle your data between epochs. You can use `shuffle_data()` from utils.py to help with this.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        # YOUR CODE HERE
        # loop through the number of epochs

        losses = []
        for epoch in range(epochs):
            # suffle the data
            x, y = shuffle_data(x, y)
            epoch_loss = 0
            # loop through the number of examples
            #print('epoch', epoch)
            for i in range(len(x)):
                
                
                # update the learning rate
                lr = self.lr0/(1+0.01*self.t)

                # increment counter
                self.t += 1

                # get the current example
                x_i = x[i]
                y_i = y[i]

                # convert the label to {-1, 1}
                if y_i == 0:
                    y_i = -1
                else:
                    y_i = 1

                #print('x_i', x_i)
                #print('y_i', y_i)

                # find the margin
                marg = y_i * (np.dot(self.w, x_i))

                # update weights and bias based on the margin
                if marg <= 1:
                    self.w += - lr * self.w + lr * self.C * y_i * x_i
                    self.b += lr * self.C * y_i
                else:
                    self.w = (1 - lr) * self.w

                epoch_loss += self.loss(x_i, y[i])
                
            losses.append(epoch_loss / len(x))

        
                
                      
               # return the final weights
        return self.w


    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        predict = []

        if x.ndim == 1:
            x = x.reshape(1, -1)


        for i in range(x.shape[0]):
            # get the current example
            x_i = x[i]
            # calculate the prediction
            prediction = np.sign(np.dot(self.w, x_i)+ self.b)
            # convert the prediction to {0, 1}
            if prediction >= 0.5:
                predict.append(1)
            else:
                predict.append(0)

        return predict

        


class LogisticRegression(Model):
    def __init__(self, num_features: int, lr0: float, sigma2: float):
        '''
        Initialize a new LogisticRegression model

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            lr0 (float): the initial learning rate (gamma_0)
            sigma2 (float): the regularization/loss tradeoff hyperparameter
        '''

        self.lr0 = lr0
        self.sigma2 = sigma2

        # YOUR CODE HERE
        

        # initialize the weights to a random number which is small to start
        self.w = np.random.rand(num_features) * 0.01
        self.b = 0
        # initialize t, the number of iterations
        self.t = 0
        # initialize the learning rate
        self.lr = lr0

        #self.plotting = True
        self.plotting = False
        


    
    def get_hyperparams(self) -> dict:
        return {'lr0': self.lr0, 'sigma2': self.sigma2}
    

    def loss(self, x_i: np.ndarray, y_i: int) -> float:
        '''
        Calculate the SVM loss on a single example.

        Args:
            x_i (np.ndarray): a 1-D np.ndarray (num_features) with the features for a single example
            y_i (int): the label for the example, either 0 or 1.

        Returns:
            float: the loss for the example using the current weights

        Hints:
            - Use the `clip()` function from utils.py to clip the input to exp() to be between -100 and 100.
                If you apply exp() to very small/large numbers, you'll likely run into a float overflow issue.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        # initialize variable
        total_loss = 0
        # find z
        z = np.dot(self.w, x_i)+self.b
        # clip z value
        z = clip(z)

        # calculate the emperical loss
        emperical_loss = np.log(1 + np.exp(-y_i *z))
        # add the regularization term
        regularization = 1/(2*self.sigma2) * np.dot(self.w, self.w)
        # add the regularization term to the loss
        total_loss = regularization + emperical_loss

        return total_loss
    
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Shuffle your data between epochs. You can use `shuffle_data()` from utils.py to help with this.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        # YOUR CODE HERE
        losses = []
        # loop through the number of epochs
        for epoch in range(epochs):
            # suffle the data
            x, y = shuffle_data(x, y)
            epoch_loss = 0
            # loop through the number of examples
            for i in range(x.shape[0]):
                # get the current example
                x_i = x[i]
                y_i = y[i]
                
                
                # convert the label to {-1, 1}
                if y_i == 0:
                    y_i = -1
                else:
                    y_i = 1

                # update the learning rate
                self.lr = self.lr0/(1+ 0.01*self.t)
                # increment counter
                self.t += 1
                
                # find z
                z = np.dot(self.w, x_i)
                # clip z value
                z = clip(z)
                
                # calculate the gradient
                gradient = -y_i * x_i * sigmoid(-y_i * z) + (1/self.sigma2) * self.w
                gradient_b = -y_i * sigmoid(-y_i * z)
                # update the weights
                self.w -= self.lr * gradient
                # update the bias
                self.b -= self.lr * gradient_b

                epoch_loss += self.loss(x_i, y[i])
                
            losses.append(epoch_loss / len(x))


        
                
        # return the final weights
        return self.w


    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        # initialize variable
        predict = []

        if x.ndim == 1:
            x = x.reshape(1, -1)


        for i in range(x.shape[0]):
            # get the current example
            x_i = x[i]
            # calculate the prediction
            prediction = np.sign(np.dot(self.w, x_i)+ self.b)
            # convert the prediction to {0, 1}
            if prediction >= 0.5:
                predict.append(1)
            else:
                predict.append(0)
        return predict


def sigmoid(z: float) -> float:
    '''
    The sigmoid function.

    Args:
        z (float): the argument to the sigmoid function.

    Returns:
        float: the sigmoid applied to z.

    Hints:
        - Use the `clip()` function from utils.py to clip the input to exp() to be between -100 and 100.
            If you apply exp() to very small/large numbers, you'll likely run into a float overflow issue.
          
    '''
    
    # YOUR CODE HERE, REMOVE THE LINE BELOW
    # clip the input to exp() to be between -100 and 100 for each value
    # to avoid overflow
    z = clip(z, max_abs_value=100)

    # initialize variable
    sigmoid = 0
    # calculate the sigmoid
    sigmoid = 1 / (1 + np.exp(-z))
    
    return sigmoid