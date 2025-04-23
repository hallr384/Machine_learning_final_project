import pandas as pd
import numpy as np

class KaggleConvertToCategory:
    def __init__(self, train_path, test_path, eval_path):
        # Load datasets
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        self.eval_data = pd.read_csv(eval_path)
        self.bins = 50000 # at a high number, this just converts data to catigorical while keeping the same values

    def preprocess(self):
        
        # Separate the label column from the rest of the data
        self.train_label = self.train_data['label']
        self.test_label = self.test_data['label']
        self.eval_label = self.eval_data['label']

        # Drop the label column from the data
        self.train_data = self.train_data.drop(columns=['label'])
        self.test_data = self.test_data.drop(columns=['label'])
        self.eval_data = self.eval_data.drop(columns=['label'])

        # Convert numerical columns to categorical by binning
        self.train_data = self.convert_to_categorical(self.train_data)
        self.test_data = self.convert_to_categorical(self.test_data)
        self.eval_data = self.convert_to_categorical(self.eval_data)

        # Convert categorical columns to numerical using one-hot encoding
        categorical_columns = self.train_data.select_dtypes(include=['object']).columns
        self.train_data = pd.get_dummies(self.train_data, columns=categorical_columns, drop_first=True)
        self.test_data = pd.get_dummies(self.test_data, columns=categorical_columns, drop_first=True)
        self.eval_data = pd.get_dummies(self.eval_data, columns=categorical_columns, drop_first=True)

        # Ensure the test and eval data have the same columns as the train data
        self.align_columns()

        # Add the label column back to the front
        self.train_data.insert(0, 'label', self.train_label)
        self.test_data.insert(0, 'label', self.test_label)
        self.eval_data.insert(0, 'label', self.eval_label)

        # Save the preprocessed data
        train_path_preprocessed = './project_data/data/train_cat.csv'
        test_path_preprocessed = './project_data/data/test_cat.csv'
        eval_path_preprocessed = './project_data/data/evalanon_cat.csv'

        self.train_data.to_csv(train_path_preprocessed, index=False)
        self.test_data.to_csv(test_path_preprocessed, index=False)
        self.eval_data.to_csv(eval_path_preprocessed, index=False)

        print("Preprocessing complete. Preprocessed data saved.")

        return train_path_preprocessed, test_path_preprocessed, eval_path_preprocessed

    def convert_to_categorical(self, df):
        """Convert numerical columns to categorical by binning with integer values."""
        for col in df.select_dtypes(include=[np.number]).columns:
            bins = self.bins
            # Create bins and assign integer labels instead of Bin_1, Bin_2, etc.
            bin_labels = np.arange(1, bins + 1)
            df[col] = pd.cut(df[col], bins=bins, labels=bin_labels, include_lowest=True)
        return df

    def align_columns(self):
        """Align the columns of the train, test, and eval data."""
        train_columns = set(self.train_data.columns)
        test_columns = set(self.test_data.columns)
        eval_columns = set(self.eval_data.columns)

        missing_in_test = train_columns - test_columns
        missing_in_train = test_columns - train_columns
        missing_in_eval = train_columns - eval_columns
        missing_in_train_eval = eval_columns - train_columns

        # Add missing columns to each dataset with NaN values for test or eval
        for col in missing_in_test:
            self.test_data[col] = np.nan
        for col in missing_in_train:
            self.train_data[col] = np.nan
        for col in missing_in_eval:
            self.eval_data[col] = np.nan
        for col in missing_in_train_eval:
            self.train_data[col] = np.nan

        # Ensure the columns are in the same order
        self.test_data = self.test_data[self.train_data.columns]
        self.eval_data = self.eval_data[self.train_data.columns]

# Create an instance of the class and call preprocess
def convert(train_path='./project_data/data/train.csv', test_path='./project_data/data/test.csv', eval_path='./project_data/data/evalanon.csv'):
    converter = KaggleConvertToCategory(train_path, test_path, eval_path)
    return converter.preprocess()
