# Perceptron 


The following libraries are needed to run this code. 

import pandas as pd
import numpy as np
import os
from glob import glob
from math import log2
from typing import Protocol, Tuple
import argparse





The file structure should be as follows

Code1_DT
    KaggleResults
        - Predictions will be save here under PER_predictions.csv

    project_data (can be downloaded from )
        - data
            - evalanon.csv
            - eval_id (txt)
            - test.csv
            - train.csv
    KagglePERdata.py
    KagglePERevaluate.py
    KagglePERmodel.py
    KagglePERtrain.py



To run this code, make sure you have added the data with proper naming convention to the folder. That data can be downloaded from https://github.com/hallr384/Machine_learning_final_project under the project_data folder. Additionally A full code can be downloaded with all data and all of the results created via my by hand cross validation at the same link under the Machine_Learning_Project_full folder.


A majority baseline was included to verify code

```sh
python KagglePERtrain.py -m majority_baseline
```

### Perceptron

For this algorithm, I implemented a decayed perceptron. To run this code, you can simply adjust the code below upon running my cross validation with learning rates 1,0.1,0.001 and epochs 1,5,10 and 20 I determined lr 0.001 with Epochs 10 gave the best training result at 0.713


```sh

# train/eval a decay perceptron with lr=1 for 10 epochs
python KagglePERtrain.py -m decay --lr 0.001 --epochs 10


```
