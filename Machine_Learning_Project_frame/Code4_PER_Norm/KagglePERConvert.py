import pandas as pd
import os
import numpy as np

class KaggleConvertToNorm:
    def __init__(self, train_path, test_path, eval_path):
        # Load datasets
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        self.eval_data = pd.read_csv(eval_path)

    def normalize_csv(self):
        # Separate labels
        train_label = self.train_data['label']
        test_label = self.test_data['label']
        eval_label = self.eval_data['label']

        # Drop label columns
        train_features = self.train_data.drop(columns=['label'])
        test_features = self.test_data.drop(columns=['label'])
        eval_features = self.eval_data.drop(columns=['label'])

        # Normalize using train set statistics
        min_vals = train_features.min()
        max_vals = train_features.max()
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Prevent division by zero

        train_norm = (train_features - min_vals) / range_vals
        test_norm = (test_features - min_vals) / range_vals
        eval_norm = (eval_features - min_vals) / range_vals

        # Add labels back
        train_norm.insert(0, 'label', train_label)
        test_norm.insert(0, 'label', test_label)
        eval_norm.insert(0, 'label', eval_label)

        # Save outputs
        train_path_preprocessed = './project_data/data/train_norm.csv'
        test_path_preprocessed = './project_data/data/test_norm.csv'
        eval_path_preprocessed = './project_data/data/evalanon_norm.csv'

        train_norm.to_csv(train_path_preprocessed, index=False)
        test_norm.to_csv(test_path_preprocessed, index=False)
        eval_norm.to_csv(eval_path_preprocessed, index=False)

        print("Preprocessing complete. Preprocessed data saved.")
        return train_path_preprocessed, test_path_preprocessed, eval_path_preprocessed

# If this script is run directly
if __name__ == "__main__":
    converter = KaggleConvertToNorm(
        train_path='./project_data/data/train.csv',
        test_path='./project_data/data/test.csv',
        eval_path='./project_data/data/evalanon.csv'
    )
    converter.normalize_csv()
