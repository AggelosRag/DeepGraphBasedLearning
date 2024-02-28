import unittest
import numpy as np
from utils import load_csv_files


class TestLoadCsvFiles(unittest.TestCase):

    lr_train_data, hr_train_data, lr_test_data = load_csv_files()
    lr_train_matrix, hr_train_matrix, lr_test_matrix = load_csv_files(
        return_matrix=True
    )

    def test_preprocessing(self):
        self.assertFalse(np.isnan(self.lr_train_data).any())
        self.assertFalse(np.isnan(self.hr_train_data).any())
        self.assertFalse(np.isnan(self.lr_test_data).any())
        self.assertFalse(np.isnan(self.lr_train_matrix).any())
        self.assertFalse(np.isnan(self.hr_train_matrix).any())
        self.assertFalse(np.isnan(self.lr_test_matrix).any())

    def test_preprocessing(self):
        self.assertFalse((self.lr_train_data < 0).any())
        self.assertFalse((self.hr_train_data < 0).any())
        self.assertFalse((self.lr_test_data < 0).any())
        self.assertFalse((self.lr_train_matrix < 0).any())
        self.assertFalse((self.hr_train_matrix < 0).any())
        self.assertFalse((self.lr_test_matrix < 0).any())

    def test_shapes_vector(self):
        self.assertEqual(self.lr_train_data.shape, (167, 12720))
        self.assertEqual(self.hr_train_data.shape, (167, 35778))
        self.assertEqual(self.lr_test_data.shape, (112, 12720))

    def test_shapes_matrix(self):
        self.assertEqual(self.lr_train_matrix.shape, (167, 160, 160))
        self.assertEqual(self.hr_train_matrix.shape, (167, 268, 268))
        self.assertEqual(self.lr_test_matrix.shape, (112, 160, 160))


if __name__ == "__main__":
    unittest.main()
