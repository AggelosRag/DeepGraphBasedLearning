"""
Main Script for Data Preprocessing, Model Training, and Prediction CSV Generation

This script includes a basic workflow for loading, preprocessing, 
training a model, and generating predictions in a machine learning task.

"""

from utils import three_fold_cross_validation, load_csv_files, save_csv_prediction
import numpy as np
import torch


np.random.seed(42)
torch.manual_seed(42)

lr_train_data, hr_train_data, lr_test_data = load_csv_files()

model = None

# Implement the model and then uncomment here...

# scores = three_fold_cross_validation(model, lr_train_data, hr_train_data)
# lr_test_predictions = model.predict(lr_test_data)
# save_csv_prediction(lr_test_predictions, logs=True)
