# Code 5 SVM

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

    project_data (can be downloaded)
        - data
            - evalanon.csv
            - eval_id (txt)
            - test.csv
            - train.csv
    KaggleSVMdata.py
    KaggleSVMevaluate.py
    KaggleSVMmodel.py
    KaggleSVMtrain.py
    utils.py


To run this code, make sure you have added the data with proper naming convention to the folder. That data can be downloaded from https://github.com/hallr384/Machine_learning_final_project under the project_data folder. Additionally A full code can be downloaded with all data and all of the results created via my by hand cross validation at the same link under the Machine_Learning_Project_full folder.




## Models

### Majority Baseline

Majority Baseline to Make sure it runs properly
```sh
python KaggleSVMtrain.py -m majority_baseline 
```

### Support Vector Machine


The SVM code can be run with the below code, you can adjust values as seen fit. The current set up gave me the best results durring my cross validation. I ran multiple iterations of the code. 

```sh
# train/eval a simple SVM at the ideal conditions.
python KaggleSVMtrain.py -m svm --lr0 0.0001 --reg_tradeoff 10 --epochs 20
```


