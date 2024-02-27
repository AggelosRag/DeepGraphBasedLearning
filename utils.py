"""
utils.py - Utility Functions

Description:
This module provides functions that may be useful for all the models used in this project, mostly related with:
pre-processing, post-processing and evaluation

"""

# Imports

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from sklearn.model_selection import KFold

import MatrixVectorizer as MV



# Functions for data loading

def load_csv_files(return_matrix = False, include_diagonal=False, logs = False):
    """
    Load CSV files and perform pre-processing.

    Parameters:
    - return_matrix (bool): If True, applies anti-vectorization to return matrices.
    - include_diagonal (bool): If True and return_matrix is True, includes diagonal elements in anti-vectorization.
    - logs (bool): If True, prints the shape of the loaded data.

    Returns:
    - Tuple of NumPy arrays: Contains loaded data (lr_train, hr_train, lr_test).
    """

    hr_train_data = np.genfromtxt("data/hr_train.csv", delimiter=',', skip_header=1)
    lr_train_data = np.genfromtxt("data/lr_train.csv", delimiter=',', skip_header=1)
    lr_test_data = np.genfromtxt("data/lr_test.csv", delimiter=',', skip_header=1)

    # Pre-processing of the values

    np.nan_to_num(hr_train_data, copy = False)
    np.nan_to_num(lr_train_data, copy = False)
    np.nan_to_num(lr_test_data, copy = False)

    hr_train_data = np.maximum(hr_train_data, 0)
    lr_train_data = np.maximum(lr_train_data, 0)
    lr_test_data = np.maximum(lr_test_data, 0)

    if return_matrix:
        # Apply anti-vectorization
        hr_train_matrixes = np.empty((hr_train_data.shape[0],268,268))
        for i,sample in enumerate(hr_train_data):
            hr_train_matrixes[i] = MV.MatrixVectorizer.anti_vectorize(sample, 268, include_diagonal)

        lr_train_matrixes = np.empty((lr_train_data.shape[0],160,160))
        for i,sample in enumerate(lr_train_data):
            lr_train_matrixes[i] = MV.MatrixVectorizer.anti_vectorize(sample, 160, include_diagonal)

        lr_test_matrixes = np.empty((lr_test_data.shape[0],160,160))
        for i,sample in enumerate(lr_test_data):
            lr_test_matrixes[i] = MV.MatrixVectorizer.anti_vectorize(sample, 160, include_diagonal)

        if logs:
            print(lr_train_matrixes.shape) # (167, 160, 160)
            print(hr_train_matrixes.shape) # (167, 268, 268)
            print(lr_test_matrixes.shape)  # (112, 160, 160)

        return (lr_train_matrixes, hr_train_matrixes, lr_test_matrixes)

    if logs:
        print(lr_train_data.shape) # (167, 12720)
        print(hr_train_data.shape) # (167, 35778)
        print(lr_test_data.shape)  # (112, 12720)

    return (lr_train_data, hr_train_data, lr_test_data)

def save_csv_prediction(input, input_matrix = False, include_diagonal=False, logs = False):
    """
    Save predictions in CSV format.

    Parameters:
    - input: Numpy array containing the predictions of test set.
    - input_matrix (bool): If True, assumes input is a adjacency matrix and applies vectorization.
    - include_diagonal (bool): If True and input_matrix is True, includes diagonal elements in vectorization.
    - logs (bool): If True, prints the shape of the vectorized matrix.

    Returns:
    - None
    """

    if input_matrix:
        # Apply vectorization
        df_vector = np.empty((input.shape[0],35778))
        for i,prediction in enumerate(input):
            df_vector[i] = MV.MatrixVectorizer.vectorize(prediction, include_diagonal)
        
        input = df_vector
        if logs:
            print(input.shape)

    meltedDF = input.to_numpy().flatten()
    file_name = "submissions/"+time.strftime("%Y%m%d-%H%M%S")+"_submission.csv"
    np.savetxt(file_name, meltedDF, delimiter=',', fmt='%f', header=["ID","Predicted"], comments='')
    if logs:
        print(f"CSV file '{file_name}' created.")


def three_fold_cross_validation(model, X, Y, random_state=None):
    """
    Perform three-fold cross-validation on a given model.

    Parameters:
    - model: The machine learning model to be evaluated.
    - X: The input features.
    - Y: The target variable.
    - random_state (int): Random seed for reproducibility (optional).

    Returns:
    - List of floats: The evaluation scores for each fold.
    """

    kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
    scores = []

    for train_index, val_index in kf.split(X):
        
        model.fit(X[train_index], Y[train_index])
        score = model.score(X[val_index], Y[val_index])
        scores.append(score)

    return scores
