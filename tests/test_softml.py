import unittest
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from softml.core.softml import evaluate_linear_regression
from softml.core.softml import evaluate_random_forest
from softml.core.softml import evaluate_polynomial_regression
from softml.core.softml import input_output_split
from softml.core.softml import soft_test_dataframe
from softml.core.softml import generate_scalers
from softml.core.softml import standardize_data, normalize_data

class SoftMLTest(unittest.TestCase):
    
    def setUp(self):

            self.random_state = 0
            np.random.seed(self.random_state)

            self.TRAIN_SIZE = 0.75

            self.df = pd.read_csv("../resources/SwedishInsuranceDataset/AutoInsurSweden_reformated.txt",\
                 sep = "\t", header = None)

            self.df_BHD = pd.read_csv("../resources/BostonHousingDataset/housing.data",\
                 delim_whitespace=True, header = None)

            self.X = self.df.iloc[:, 0].values.reshape(-1, 1)
            self.Y = self.df.iloc[:, 1].values.reshape(-1, 1)

            self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X, self.Y, train_size = self.TRAIN_SIZE, random_state = self.random_state)

            # Standardization & Normalization
            self.scalers = generate_scalers(self.X_train, self.Y_train)

            self.X_train_standardized, self.X_val_standardized, self.Y_train_standardized = standardize_data(self.X_train, self.X_val, self.Y_train, self.scalers)
            self.X_train_normalized, self.X_val_normalized, self.Y_train_normalized = normalize_data(self.X_train, self.X_val, self.Y_train, self.scalers)

    def test_evaluate_linear_regression(self) :

        # MSE with no scaler
        mse = evaluate_linear_regression(self.X_train, self.Y_train, self.X_val, self.Y_val)
        self.assertAlmostEqual(mse, 1499.5468481994199)

        # MSE with standardization
        mse = evaluate_linear_regression(self.X_train_standardized, self.Y_train_standardized, self.X_val_standardized, self.Y_val, scaler = self.scalers["StandardScaler"]["Y"])
        self.assertAlmostEqual(mse, 1499.5468481994199)

        # MSE with normalization
        mse = evaluate_linear_regression(self.X_train_normalized, self.Y_train_normalized, self.X_val_normalized, self.Y_val, scaler = self.scalers["MinMaxScaler"]["Y"])
        self.assertAlmostEqual(mse, 1499.5468481994199)

    def test_evaluate_polynomial_regression(self) :

        # MSE with no scaler - degree = 2
        mse = evaluate_polynomial_regression(
            self.X_train, self.Y_train, self.X_val, self.Y_val, 
            degree = 2)
        self.assertAlmostEqual(mse, 1441.9536333497103)

        # MSE with standardization - degree = 2
        mse = evaluate_polynomial_regression(
            self.X_train_standardized, self.Y_train_standardized, self.X_val_standardized, self.Y_val, 
            degree = 2, scaler = self.scalers["StandardScaler"]["Y"])
        self.assertAlmostEqual(mse, 1441.9536333497101)

        # MSE with standardization - degree = 3
        mse = evaluate_polynomial_regression(
            self.X_train_normalized, self.Y_train_normalized, self.X_val_normalized, self.Y_val, 
            degree = 3, scaler = self.scalers["MinMaxScaler"]["Y"])
        self.assertAlmostEqual(mse, 1377.0062199850454)

    def test_evaluate_random_forest(self) :

        mse = evaluate_random_forest(self.X_train, self.Y_train, self.X_val, self.Y_val, random_state = self.random_state)
        self.assertAlmostEqual(mse, 1849.968935913339)

    def test_input_output_split(self) :

        X, Y = input_output_split(self.df, 1)

        self.assertEqual(X[0], self.X[0])
        self.assertEqual(X[5], self.X[5])
        self.assertEqual(Y[10], self.Y[10])

        self.assertEqual(X.shape, self.X.shape)
        self.assertEqual(Y.shape, self.Y.shape)

    def test_soft_test_dataframe_integrity(self) : 

        # Guarantee integrity between test_soft_test_dataframe & scikit results

        response = soft_test_dataframe(self.df, 1, train_size = self.TRAIN_SIZE, random_state = self.random_state)
        mse_LR = evaluate_linear_regression(self.X_train, self.Y_train, self.X_val, self.Y_val)
        mse_RF = evaluate_random_forest(self.X_train, self.Y_train, self.X_val, self.Y_val, random_state = self.random_state)

        self.assertAlmostEqual(response["LinearRegression"]["StandardScaler"]["mse"], 1499.5468481994199)
        self.assertAlmostEqual(response["LinearRegression"]["MinMaxScaler"]["mse"], 1499.5468481994199)
        self.assertAlmostEqual(response["LinearRegression"]["NoScaler"]["mse"], 1499.5468481994199)
        self.assertAlmostEqual(response["RandomForest"]["NoScaler"]["mse"], mse_RF)

    def test_soft_test_dataframe_performance(self) :

        # Performance validation on BostonHousingDataset

        # Performance provided by
        # https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
        # is MSE +/-= 24 - 27 (NN)

        # Performance provided by
        # https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155
        # is MSE +/-= 26

        response = soft_test_dataframe(self.df_BHD, 13, train_size = self.TRAIN_SIZE, random_state = self.random_state)

        self.assertAlmostEqual(response["LinearRegression"]["StandardScaler"]["mse"], 29.782245092302368)
        self.assertAlmostEqual(response["LinearRegression"]["MinMaxScaler"]["mse"], 29.782245092302432)
        self.assertAlmostEqual(response["RandomForest"]["NoScaler"]["mse"], 16.726365055118123)
                         
if __name__ == "__main__" :
    unittest.main()