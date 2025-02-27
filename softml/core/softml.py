import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

from softml.util.utils import nested_dictionnary

def evaluate_linear_regression(X_train, Y_train, X_val, Y_val, scaler = None) :

    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train, Y_train)
    Y_pred = linear_regressor.predict(X_val)

    if scaler :
        Y_pred = scaler.inverse_transform(Y_pred) 

    return(mean_squared_error(Y_pred, Y_val))

def evaluate_polynomial_regression(X_train, Y_train, X_val, Y_val, degree = 2, scaler = None) :

    polynomial_features= PolynomialFeatures(degree = degree, include_bias = False)
    X_train_poly = polynomial_features.fit_transform(X_train)
    X_val_poly = polynomial_features.fit_transform(X_val)

    linear_regressor = LinearRegression()

    linear_regressor.fit(X_train_poly, Y_train)
    Y_pred = linear_regressor.predict(X_val_poly)

    if scaler :
        Y_pred = scaler.inverse_transform(Y_pred) 

    return(mean_squared_error(Y_pred, Y_val))

def evaluate_random_forest(X_train, Y_train, X_val, Y_val, random_state = None) :

    if random_state is not None :
        np.random.seed(random_state)

    random_forest_regressor = RandomForestRegressor() 
    random_forest_regressor.fit(X_train, Y_train.ravel())
    Y_pred = random_forest_regressor.predict(X_val)

    return(mean_squared_error(Y_pred, Y_val))

def input_output_split(df, Y_columns) : 

    Y_columns = [Y_columns] if (isinstance(Y_columns, str) or isinstance(Y_columns, int)) else Y_columns

    Y = df[Y_columns].values
    X = df.drop(Y_columns, axis = 1).values

    return X, Y

def generate_scalers(X_train, Y_train) :

    response = {}

    nested_dictionnary(response, ["StandardScaler","X"], StandardScaler().fit(X_train))
    nested_dictionnary(response, ["StandardScaler","Y"], StandardScaler().fit(Y_train))
    nested_dictionnary(response, ["MinMaxScaler","X"], MinMaxScaler().fit(X_train))
    nested_dictionnary(response, ["MinMaxScaler","Y"], MinMaxScaler().fit(Y_train))

    return response

def standardize_data(X_train, X_val, Y_train, scalers) :

    X_train_standardized = scalers["StandardScaler"]["X"].transform(X_train)
    X_val_standardized   = scalers["StandardScaler"]["X"].transform(X_val)
    Y_train_standardized = scalers["StandardScaler"]["Y"].transform(Y_train)

    return(X_train_standardized, X_val_standardized, Y_train_standardized)

def normalize_data(X_train, X_val, Y_train, scalers) :

    X_train_normalized = scalers["MinMaxScaler"]["X"].transform(X_train)
    X_val_normalized   = scalers["MinMaxScaler"]["X"].transform(X_val)
    Y_train_normalized = scalers["MinMaxScaler"]["Y"].transform(Y_train)

    return(X_train_normalized, X_val_normalized, Y_train_normalized)

def soft_test_dataframe(df, Y_columns, polynomial_regression_degree = 2, train_size = 0.75, random_state = None) :

    response = {}

    X, Y = input_output_split(df, Y_columns)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size = train_size, random_state = random_state)

    scalers = generate_scalers(X_train, Y_train)

    X_train_standardized, X_val_standardized, Y_train_standardized = standardize_data(X_train, X_val, Y_train, scalers)
    X_train_normalized, X_val_normalized, Y_train_normalized = normalize_data(X_train, X_val, Y_train, scalers)

    #####
    #
    # LINEAR REGRESSION
    #
    #####

    mse_LR = evaluate_linear_regression(
        X_train, Y_train, X_val, Y_val, 
        scaler = None)
    nested_dictionnary(response, ["LinearRegression","NoScaler", "mse"], mse_LR)

    mse_LR_standardized = evaluate_linear_regression(
        X_train_standardized, Y_train_standardized, X_val_standardized, Y_val, 
        scaler = scalers["StandardScaler"]["Y"])
    nested_dictionnary(response, ["LinearRegression","StandardScaler", "mse"], mse_LR_standardized)

    mse_LR_normalized = evaluate_linear_regression(
        X_train_normalized, Y_train_normalized, X_val_normalized, Y_val, 
        scaler = scalers["MinMaxScaler"]["Y"])
    nested_dictionnary(response, ["LinearRegression","MinMaxScaler", "mse"], mse_LR_normalized)

    #####
    #
    # POLYNOMIAL REGRESSION
    #
    #####

    for degree in range(2, polynomial_regression_degree+1, 1) :

        mse_PR = evaluate_polynomial_regression(
            X_train, Y_train, X_val, Y_val, 
            scaler = None,
            degree = degree)
        nested_dictionnary(response, ["PolynomialRegression","NoScaler", f"degree = {degree}","mse"], mse_PR)

        mse_PR_standardized = evaluate_polynomial_regression(
            X_train_standardized, Y_train_standardized, X_val_standardized, Y_val, 
            scaler = scalers["StandardScaler"]["Y"],
            degree = degree)
        nested_dictionnary(response, ["PolynomialRegression","StandardScaler", f"degree = {degree}", "mse"], mse_PR_standardized)

        mse_PR_normalized = evaluate_polynomial_regression(
            X_train_normalized, Y_train_normalized, X_val_normalized, Y_val, 
            scaler = scalers["MinMaxScaler"]["Y"],
            degree = degree)
        nested_dictionnary(response, ["PolynomialRegression","MinMaxScaler", f"degree = {degree}", "mse"], mse_PR_normalized)

    #####
    #
    # RANDOM FOREST
    #
    #####

    mse_RF = evaluate_random_forest(
        X_train, Y_train, X_val, Y_val,
        random_state = random_state)
    nested_dictionnary(response, ["RandomForest","NoScaler", "mse"], mse_RF)


    return response



if __name__ == "__main__" :
	pass