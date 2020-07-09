from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np

# TODO : do normalization ?

def evaluate_linear_regression(X_train, Y_train, X_val, Y_val) :

    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train, Y_train)
    Y_pred = linear_regressor.predict(X_val) 

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
	
def soft_test_dataframe(df, Y_columns, train_size = 0.75, random_state = None) :

    X, Y = input_output_split(df, Y_columns)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size = train_size, random_state = random_state)

    mse1 = evaluate_linear_regression(X_train, Y_train, X_val, Y_val)
    mse2 = evaluate_random_forest(X_train, Y_train, X_val, Y_val, random_state = random_state)

    return({
        "RandomForest" : {"mse" : mse2},
        "LinearRegression" : {"mse" : mse1},
        })



if __name__ == "__main__" :
	pass