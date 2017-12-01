#!/usr/bin/python
import numpy as np
import math
NUM_FEAT = 95
# x: n x (p+1) matrix
# y: n x 1 vector
# Train weights via closed form of linear regression
def train_linear_regression(x, y):
    x_t = np.transpose(x)
    gram = np.dot(x_t, x)
    gram_inv = np.linalg.inv(gram)
    x_t_y = np.dot(x_t, y)
    w = np.dot(gram_inv, x_t_y)

    return w

# Train weights via linear regression with gradient descent
def train_linear_gradient_descent(x, y, learning_rate):
    thresh = 10E-9  # Could work larger, but this is safer
    x_t = np.transpose(x)
    w_old = np.random.normal(size=(NUM_FEAT + 1, 1))
    w_new = np.random.normal(size=(NUM_FEAT + 1, 1))
    # Calculate difference between guesses
    diff = np.amax(np.absolute(w_new - w_old))
    steps = 0
    # Go until difference is close enough to 0
    while diff > thresh:
        steps += 1

        w_old = w_new
        diff = (y - np.dot(x, w_old))
        gradient = np.dot(x_t, diff)
        w_new = w_old + learning_rate * gradient

        # Break if w diverges
        if np.isnan(w_new - w_old - thresh).any():
            print "DIVERGED"
            break
        diff = np.amax(np.absolute(w_new - w_old))

    print "steps =", steps
    return w_new

# l: hyperparameter
# K-folds Cross validation with k=5 for ridge regression
def ridge_k_folds(x, y, l):
    # 5-fold cross validation
    div_size = len(x) / 5
    error_results = []
    x_split = np.split(x, 5)
    y_split = np.split(y, 5)
    # Form all combinations of training data sets
    for i in range(5):
        validation_x = x_split[i]
        validation_y = y_split[i]

        train_x = []
        train_y = []
        for j in range(5):
            if i != j:
                train_x.append(x_split[j])
                train_y.append(y_split[j])
        t_x = np.concatenate(train_x)
        t_y = np.concatenate(train_y)

        # Calculate w from current fold
        w = train_ridge_regression(t_x, t_y, l)
        # Test w against set not in fold
        error = get_RMSE(validation_x, validation_y, w)
        error_results.append(error)

    return sum(error_results) / len(error_results)

# Train weights via closed form of ridge regression
def train_ridge_regression(x, y, l):
    x_t = np.transpose(x)
    gram = np.dot(x_t, x)
    I = np.identity(NUM_FEAT + 1, dtype="float")
    ridge = gram + (l * I)
    ridge_inv = np.linalg.inv(ridge)
    x_t_y = np.dot(x_t, y)
    w = np.dot(ridge_inv, x_t_y)

    return w

# Train weights via ridge regression with gradient descent
def train_ridge_gradient_descent(x, y, l, learning_rate):
    thresh = 10E-9  # Could work larger, but this is safer
    x_t = np.transpose(x)
    w_old = np.random.normal(size=(NUM_FEAT + 1, 1))
    w_new = np.random.normal(size=(NUM_FEAT + 1, 1))
    # Calculate difference between guesses
    diff = np.amax(np.absolute(w_new - w_old))
    steps = 0
    # Go until difference is close enough to 0
    while diff > thresh:
        steps += 1

        w_old = w_new
        diff = (y - np.dot(x, w_old))
        gradient = np.dot(x_t, diff)
        ridge = gradient - (l * w_old)
        w_new = w_old + learning_rate * ridge
        # Break if w diverges
        if np.isnan(w_new - w_old - thresh).any():
            print "DIVERGED"
            break
        diff = np.amax(np.absolute(w_new - w_old))
    print "steps =", steps
    return w_new

# Calculate root mean squared error for trained w
def get_RMSE(x, y, w):
    accum = 0
    predict = lambda x, w: np.dot(np.transpose(w), x)

    for i in range(len(x)):
        prediction = predict(x[i], w)
        accum += MSE(prediction, y[i])

    mean_squared_error = accum / (i + 1)
    # Root mean squared error
    RMSE = math.sqrt(mean_squared_error)

    return RMSE
# Mean squared error
def MSE(prediction, actual):
    m = len(actual)
    MSE = ((prediction - actual)**2) / m
    return MSE

# Process data into useful components
def load_data(fname):
    data = np.loadtxt(fname, delimiter="\t", dtype="str")
    # Drop header row
    data = data[1:]
    # First column is label
    y = data[:, 0].astype("float")
    # Quick way to select all columns except the first in numpy
    x = data[:, 1:].astype("float")
    # Add artificial feature (column vector of 1s) to account for b in weights vector
    artificial = np.ones(shape=(x.shape[0], 1), dtype="float")
    x = np.append(x, artificial, axis=1)

    return x, y

if __name__ == "__main__":
    x_train, y_train = load_data("crime-train.txt")
    x_test, y_test = load_data("crime-test.txt")
    # Explicitly define vectors as column vectors
    y_test.shape = (399, 1)
    y_train.shape = (1595, 1)

    ### CLOSED FORM ###
    ## Linear regression ##
    w = train_linear_regression(x_train, y_train)
    print "training:", get_RMSE(x_train, y_train, w)
    print "testing:", get_RMSE(x_test, y_test, w)
    ## Ridge regression ##
    # K Folds #
    l = 400.00
    for i in range(10):
        print "lambda=", l, "err =", ridge_k_folds(x_train, y_train, l)
        l = l/2
    # By inspection, the above block implies lambda = 25 is ideal (lowest RMSE)
    w = train_ridge_regression(x_train, y_train, 25)
    print "testing:", get_RMSE(x_test, y_test, w)

    ### GRADIENT DESECENT ###
    ## Linear Regression ##
    w = train_linear_gradient_descent(x_train, y_train, 10E-6)
    print "training:", get_RMSE(x_train, y_train, w)
    print "testing:", get_RMSE(x_test, y_test, w)
    ## Ridge Regression ##
    w = train_ridge_gradient_descent(x_train, y_train, 25, 10E-6)
    print "testing:", get_RMSE(x_test, y_test, w)
