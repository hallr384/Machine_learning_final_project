''' This file defines the model classes that will be used. 
    You need to add your code wherever you see "YOUR CODE HERE".
'''

from typing import Protocol, Tuple

import numpy as np

# set the numpy random seed so our randomness is reproducible
np.random.seed(1)


# DON'T CHANGE THE CLASS BELOW! 
# You will implement the train and predict functions in the Perceptron classes further down.
class Model(Protocol):
    def get_hyperparams(self) -> dict:
        ...
        
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        ...

    def predict(self, x: np.ndarray) -> list:
        ...


class MajorityBaseline(Model):
    def __init__(self):
        
        # YOUR CODE HERE, REMOVE THE LINE BELOW
        #initialize variables
        self.MostCommonLabel = None


        pass


    def get_hyperparams(self) -> dict:
        return {}
    

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


class Perceptron(Model):
    def __init__(self, num_features: int, lr: float, decay_lr: bool = False, mu: float = 0):
        '''
        Initialize a new Perceptron.

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            lr (float): the learning rate (eta). This is also the initial learning rate if decay_lr=True
            decay_lr (bool): whether or not to decay the initial learning rate lr
            mu (float): the margin (mu) that determines the threshold for a mistake. Defaults to 0
        '''     

        self.lr = lr
        self.decay_lr = decay_lr
        self.mu = mu
        self.starting_lr = lr

        # YOUR CODE HERE

        # initialize weight and bias to be small random numbers between -0.01 and 0.01
        self.w = np.random.uniform(-0.0001, 0.0001, num_features)
        self.b = np.random.uniform(-0.0001, 0.0001)

        # initialize t for the number of updates
        self.t = 0

        # initialize the number of updates that have been made
        self.num_updates = 0
        

        
        

    def get_hyperparams(self) -> dict:
        return {'lr': self.lr, 'decay_lr': self.decay_lr, 'mu': self.mu}
    
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Remember to shuffle your data between epochs.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        # YOUR CODE HERE

        # loop through the epochs
        for epoch in range(epochs):
            # shuffle the data
            x, y = shuffle_data(x, y)
            
            # loop through each example
            for i in range(len(x)):
                # grab the example value and label
                x_i = x[i]
                y_i = y[i]

                # print the example
                #print(f"Example {i}:  Label: {y_i}")
                # map yi to -1 or 1
                if y_i == 0:
                    y_i_calc = -1
                else:
                    y_i_calc = 1

                # predict what the label should be
                y_pred = Perceptron.predict(self, x_i)

 

               
                # if the prediction is wrong, update the weights
                # print wrong
                #print(f"Prediction: {y_pred}, Actual: {y_i}")
                if y_pred != y_i:
                    #print('prediction wrong')
                    self.w += self.lr * y_i_calc * x_i
                    self.b += self.lr * y_i_calc
                    # update the number of updates
                    self.num_updates += 1
                else:
                    #print('prediction correct')
                    # update the number of updates
                    pass
               
                    

                
                # update the learning rate if decay is true
                if self.decay_lr== "True":
                    self.lr = self.starting_lr / (1 + self.t)
                    
                


                # increment t
                self.t += 1
        # print the total number of updates
        print(f"Total number of updates: {self.num_updates}")

                


                
        
    

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
 
        y_pred = np.sign(np.dot(x, self.w) + self.b)
        
        if y_pred == 1:
            y_pred = 1
        # map -1 to 0
        elif y_pred == -1:
            y_pred = 0


        
        

        return y_pred
    

class AveragedPerceptron(Model):
    def __init__(self, num_features: int, lr: float):
        '''
        Initialize a new AveragedPerceptron.

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            lr (float): the learning rate eta
        '''     

        self.lr = lr
        
        # YOUR CODE HERE
        # initialize weight and bias to be small random numbers between -0.01 and 0.01
        np.random.seed(42)
        self.w = np.random.uniform(-0.01, 0.01, num_features)
        self.b = np.random.uniform(-0.01, 0.01)

        # initialize t for the number of updates
        self.t = 0

        # initialize the average weights and bias
        self.a = np.zeros(num_features)
        self.b_avg = 0

        # initialize the number of updates
        self.num_updates = 0

    def get_hyperparams(self) -> dict:
        return {'lr': self.lr}
    
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Remember to shuffle your data between epochs.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        # YOUR CODE HERE
        # loop through the epochs
        for epoch in range(epochs):
            # shuffle the data
            x, y = shuffle_data(x, y)

            # loop through each example
            for i in range(len(x)):
                # grab the example value and label
                x_i = x[i]
                y_i = y[i]

                # predict what the label should be
                y_pred = Perceptron.predict(self, x_i)

                # if the prediction is wrong, update the weights
                if y_pred != y_i:
                    self.w += self.lr * y_i * x_i
                    self.b += self.lr * y_i
                    # update the number of updates
                    self.num_updates += 1
                    
            
                
                # add the weights to the total weights
                self.a = self.a + self.w
                self.b_avg = self.b_avg + self.b

        
        # print the total number of updates
        print(f"Total number of updates: {self.num_updates}")
        



    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        # calculate the labeled prediction
        y_pred = np.sign(np.dot(x, self.a) + self.b_avg)
        
        return y_pred
    

class AggressivePerceptron(Model):
    def __init__(self, num_features: int, mu: float):
        '''
        Initialize a new AggressivePerceptron.

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            mu (float): the hyperparameter mu
        '''     

        self.mu = mu
        
        # YOUR CODE HERE
        # initialize weight and bias to be small random numbers between -0.01 and 0.01
        self.w = np.random.uniform(-0.01, 0.01, num_features)
        self.b = np.random.uniform(-0.01, 0.01)

        # initialize t for the number of updates
        self.t = 0

        # initialize the number of updates
        self.num_updates = 0

    def get_hyperparams(self) -> dict:
        return {'mu': self.mu}
    
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Remember to shuffle your data between epochs.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        # YOUR CODE HERE
        # Determine first learning rate
        # loop through the epochs
        for epoch in range(epochs):
            # shuffle the data
            x, y = shuffle_data(x, y)

            # loop through each example
            for i in range(len(x)):
                # grab the example value and label
                x_i = x[i]
                y_i = y[i]
                # update the learning rate
                self.lr = (self.mu - y_i * (np.dot(x_i, self.w) + self.b)) / (np.linalg.norm(x_i) ** 2 + 1)

                # predict what the label should be
                y_pred = Perceptron.predict(self, x_i)

                # if the prediction is wrong, update the weights
                if y_pred != y_i:
                    self.w += self.lr * y_i * x_i
                    self.b += self.lr * y_i
                    # update the number of updates
                    self.num_updates += 1
                    
                # update the learning rate for aggressive margin
                if y_i * (np.dot(x_i, self.w) + self.b) < self.mu:
                    # update weights
                    self.w += self.lr * y_i * x_i
                    self.b += self.lr * y_i
                    
        # print the total number of updates
        print(f"Total number of updates: {self.num_updates}")

  
    

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        y_pred = np.sign(np.dot(x, self.w) + self.b)

        return y_pred


# DON'T MODIFY THE FUNCTIONS BELOW!
PERCEPTRON_VARIANTS = ['simple', 'decay', 'margin', 'averaged', 'aggressive']
MODEL_OPTIONS = ['majority_baseline'] + PERCEPTRON_VARIANTS
def init_perceptron(variant: str, num_features: int, lr: float, mu: float) -> Model:
    '''
    This is a helper function to help you initialize the correct variant of the Perceptron

    Args:
        variant (str): which variant of the perceptron to use. See PERCEPTRON_VARIANTS above for options
        num_features (int): the number of features (i.e. dimensions) the model will have
        lr (float): the learning rate hyperparameter eta. Same as initial learning rate for decay setting
        mu (float): the margin hyperparamter mu. Ignored for variants "simple", "decay", and "averaged"

    Returns
        (Model): the initialized perceptron model
    '''
    
    assert variant in PERCEPTRON_VARIANTS, f'{variant=} must be one of {PERCEPTRON_VARIANTS}'

    if variant == 'simple':
        return Perceptron(num_features=num_features, lr=lr, decay_lr=False)
    elif variant == 'decay':
        return Perceptron(num_features=num_features, lr=lr, decay_lr=True)
    elif variant == 'margin':
        return Perceptron(num_features=num_features, lr=lr, decay_lr=True, mu=mu)
    elif variant == 'averaged':
        return AveragedPerceptron(num_features=num_features, lr=lr)
    elif variant == 'aggressive':
        return AggressivePerceptron(num_features=num_features, mu=mu)


def shuffle_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Helper function to shuffle two np.ndarrays s.t. if x[i] <- x[j] after shuffling,
    y[i] <- y[j] after shuffling for all i, j.

    Args:
        x (np.ndarray): the first array
        y (np.ndarray): the second array

    Returns
        (np.ndarray, np.ndarray): tuple of shuffled x and y
    '''

    assert len(x) == len(y), f'{len(x)=} and {len(y)=} must have the same length in dimension 0'
    p = np.random.permutation(len(x))
    return x[p], y[p]
