# DecisionTree with Binned data

The following libraries are needed to run this code. 

import pandas as pd
import numpy as np
import os
from glob import glob
from math import log2
from typing import Protocol



The file structure should be as follows

Code1_DT
    KaggleResults
        - Predictions will be save here under DT_predictions.csv

    project_data (can be downloaded from )
        - data
            - evalanon.csv
            - eval_id (txt)
            - test.csv
            - train.csv
    KaggleConvertToCategory.py
    KaggleDTdata.py
    KaggleDTmodel.py
    KaggleDTtrain.py


To run this code, make sure you have added the data with proper naming convention to the folder. That data can be downloaded from https://github.com/hallr384/Machine_learning_final_project under the project_data folder. Additionally A full code can be downloaded with all data and all of the results created via my by hand cross validation at the same link under the Machine_Learning_Project_full folder.

'''sh

'''




This is the base model for the decision tree algorithm, 
Verification of functionality comes from the magority_baseline code

```sh
python KaggleDTtrain.py -m majority_baseline
```


Once that has been done, then the decision tree can be evaluated at various depths. I ran the code at depths 1-7 with various bin sizes ranging from 10 to 300 and dermined that depth 1 with 15000 bins gave the best percentage accuarcy on the test set at 0.734. To run that code for yourself adjust bin size in the init collumn of KaggleConvertToCategory.py and then run the code below

Make sure to adjust bin size in KaggleConvertToCategory.py on line 10

self.bins = 1500 

```sh
python KaggleDTtrain.py -m decision_tree -d 1             
```

It wil save the outcome as DT_predictions.csv Which can be renamed later if wanting to keep the results

