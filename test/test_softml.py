import sys
sys.path.append("../softml/")

import unittest
import random
import softml

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split



class SoftMLTest(unittest.TestCase):
    
    def setUp(self):

            self.random_state = 0
            np.random.seed(self.random_state)

            self.TRAIN_SIZE = 0.75

            self.df = pd.read_csv("../resources/SwedishInsuranceDataset/AutoInsurSweden_reformated.txt",\
                 sep = "\t", header = None)

            self.X = self.df.iloc[:, 0].values.reshape(-1, 1)
            self.Y = self.df.iloc[:, 1].values.reshape(-1, 1)

            self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X, self.Y, train_size = self.TRAIN_SIZE, random_state = self.random_state)

    def test_evaluate_linear_regression(self) :

        mse = softml.evaluate_linear_regression(self.X_train, self.Y_train, self.X_val, self.Y_val)
        self.assertAlmostEqual(mse, 1499.5468481994199)

    def test_evaluate_random_forest(self) :

        mse = softml.evaluate_random_forest(self.X_train, self.Y_train, self.X_val, self.Y_val, random_state = self.random_state)
        self.assertAlmostEqual(mse, 1849.968935913339)

    def test_input_output_split(self) :

        X, Y = softml.input_output_split(self.df, 1)

        self.assertEqual(X[0], self.X[0])
        self.assertEqual(X[5], self.X[5])
        self.assertEqual(Y[10], self.Y[10])

        self.assertEqual(X.shape, self.X.shape)
        self.assertEqual(Y.shape, self.Y.shape)

    def test_soft_test_dataframe(self) : 

        response = softml.soft_test_dataframe(self.df, 1, train_size = self.TRAIN_SIZE, random_state = self.random_state)
        mse_LR = softml.evaluate_linear_regression(self.X_train, self.Y_train, self.X_val, self.Y_val)
        mse_RF = softml.evaluate_random_forest(self.X_train, self.Y_train, self.X_val, self.Y_val, random_state = self.random_state)

        self.assertAlmostEqual(response["LinearRegression"]["mse"], mse_LR)
        self.assertAlmostEqual(response["RandomForest"]["mse"], mse_RF)

                         
if __name__ == "__main__" :
    unittest.main()