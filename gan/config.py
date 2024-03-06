"""
This file contains the Args class, which is used to store the arguments for the AGSR model.
"""


import constants


class Args:
    def __init__(self):
        # Define default arguments
        self.epochs = 1000
        self.lr = 0.0001
        self.lr_dim = constants.LR_MATRIX_SIZE
        self.hr_dim = constants.HR_MATRIX_SIZE
        self.hidden_dim = 512
        self.mean_gaussian = 0.0
        self.std_gaussian = 0.1
        self.device = None
        self.min_epochs = 20
        self.grace_period = 10
        self.normalisation_function = lambda x: x
        self.init_x_method = "eye"
        self.ks = [0.9, 0.7, 0.6, 0.5]
        self.k = 3
        self.alpha = 0.5
