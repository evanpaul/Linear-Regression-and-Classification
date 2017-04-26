#!/usr/bin/python
# Homework 2: QDA & LDA
# Evan Paul
import math
import numpy as np

# Process data into three matrices (split into training and testing) corresponding to classification
def process_data_set(use_cols=(0,1,2,3)):
    # Parameters columns: x1, x2, x3, x4
    print "Using parameters:", use_cols
    a = np.genfromtxt("iris.data", delimiter=",", usecols=use_cols, unpack=True, dtype=float)

    # Unpack the tuple into an array
    cols = []
    for i in range(num_feat):
        cols.append(a[i])

    # Iris-setosa
    setosa_train_rows = []
    setosa_test_rows = []
    for i in range(50):
        row = [cols[q][i] for q in range(num_feat)]

        if i < 40:
            setosa_train_rows.append(row)
        else:
            setosa_test_rows.append(row)

    setosa_train = np.asarray(setosa_train_rows)
    setosa_test = np.asarray(setosa_test_rows)

    # Iris-versicolor
    versicolor_train_rows = []
    versicolor_test_rows = []
    for j in range(50, 100):
        row = [cols[p][j] for p in range(num_feat)]

        if j < 90:
            versicolor_train_rows.append(row)
        else:
            versicolor_test_rows.append(row)

    versicolor_train = np.asarray(versicolor_train_rows)
    versicolor_test = np.asarray(versicolor_test_rows)

    # Iris-virginica
    virginica_train_rows = []
    virginica_test_rows = []
    for k in range(100, 150):
        row = [cols[r][k] for r in range(num_feat)]

        if k < 140:
            virginica_train_rows.append(row)
        else:
            virginica_test_rows.append(row)

    virginica_train = np.asarray(virginica_train_rows)
    virginica_test = np.asarray(virginica_test_rows)

    return {"setosa": {"training": setosa_train, "testing": setosa_test},
            "versicolor": {"training": versicolor_train, "testing": versicolor_test},
            "virginica": {"training": virginica_train, "testing": virginica_test}}
# x: parameter 2D array
def train(x):
    x = np.asmatrix(x)  # Convert to facilitate matrix math
    # Estimate mu (arithmetic mean)
    mu = np.array([0.0 for z in range(num_feat)])
    for i in range(num_feat):
        col = x[:, i]  # ith column vector
        mu[i] = math.fsum(col) / len(col)

    # Estimate sigma (covariance) REVIEW
    sigma = np.zeros(shape=(num_feat, num_feat))
    for j in range(len(x)):
        a = x[j] - mu
        b = np.transpose(x[j] + mu)
        outer_product = np.outer(a, b)  # outer product

        sigma += outer_product
    sigma = sigma / len(x)

    return mu, sigma
# Gaussian probability density function
def gaussian_pdf(x, mu, sigma):
    # Just to be explicit that we're dealing with column vectors in this formulas
    x.shape = (num_feat, 1)
    mu.shape = (num_feat, 1)
    # The constant term isn't included since it isn't necessary

    a = -0.5 * np.transpose(x - mu)
    b = np.asmatrix(a) * np.linalg.inv(sigma) #?
    c = b * np.asmatrix(x - mu) #?

    prob = (np.linalg.det(sigma) ** -0.5) * math.exp(c)

    return prob
# Classify input vector x into one of the three classes using Quadratic Discriminant Analysis (QDA) by default
# LDA=True to use Linear Discriminant Analysis instead
def classify(x, LDA_flag, force_diagonal=False):
    global sigma1, sigma2, sigma3
    if LDA_flag:
        avg_sigma = (sigma1 + sigma2 + sigma3)/3
        # Force sigma to be a diagonal matrix
        if force_diagonal:
            # TODO: Figure this out
            for i in range(len(avg_sigma)):
                for j in range(len(avg_sigma[i])):
                    if i != j:
                        avg_sigma[i][j] = 0
            # avg_sigma = np.diag(np.diag(avg_sigma))
        c1 = gaussian_pdf(x, mu1, avg_sigma)
        c2 = gaussian_pdf(x, mu2, avg_sigma)
        c3 = gaussian_pdf(x, mu3, avg_sigma)
    else:
        # Force sigma to be a diagonal matrix (all sigmas should be same shape)
        if force_diagonal:
            y = np.diag(np.diag(sigma1))
            for i in range(len(sigma1)):
                for j in range(len(sigma1[i])):
                    if i != j:
                        sigma1[i][j] = 0
                        sigma2[i][j] = 0
                        sigma3[i][j] = 0
            # print "sigma1"
            # print sigma1
            # print "y"
            # print y
            # sigma1 = np.diag(np.diag(sigma1))
            # sigma2 = np.diag(np.diag(sigma2))
            # sigma3 = np.diag(np.diag(sigma1))

        c1 = gaussian_pdf(x, mu1, sigma1)
        c2 = gaussian_pdf(x, mu2, sigma2)
        c3 = gaussian_pdf(x, mu3, sigma3)

    c = max(c1, c2, c3)

    if c == c1:
        iris = "setosa"
    elif c == c2:
        iris = "versicolor"
    elif c == c3:
        iris = "virginica"
    else:
        iris = "NULL"

    return iris

def evaluate_model(LDA_flag, force_diagonal=False):
    print "=" * 60 # For clarity in stdout
    if LDA_flag:
        print "Classifying data using linear discriminant analysis"
    else:
        print "Classifying data using quadratic discriminant analysis"
    if force_diagonal:
        print "Forcing covariance matrix (sigma) to be a diagonal matrix"
    # Any errors (in LDA) means the data is not linearly separable
    setosa_err = False
    versicolor_err = False
    virginica_err = False

    # Training data accuracy (less important)
    training_tries = 0.0
    training_wrong = 0.0
    for i in range(40):
        result = classify(data["setosa"]["training"][i], LDA_flag, force_diagonal)
        training_tries += 1
        if result != "setosa":
            training_wrong += 1
            setosa_err = True
    for j in range(40):
        result = classify(data["versicolor"]["training"][j], LDA_flag, force_diagonal)
        training_tries += 1
        if result != "versicolor":
            training_wrong += 1
            versicolor_err = True
    for k in range(40):
        result = classify(data["virginica"]["training"][k], LDA_flag, force_diagonal)
        training_tries += 1
        if result != "virginica":
            training_wrong += 1
            virginica_err = True

    training_error_rate = training_wrong/training_tries
    print "TRAINING ERROR RATE=", training_error_rate * 100, "%"
    # Testing data accuracy (more important)
    testing_tries = 0.0
    testing_wrong = 0.0
    for i in range(10):
        result = classify(data["setosa"]["testing"][i], LDA_flag, force_diagonal)
        testing_tries += 1
        if result != "setosa":
            testing_wrong += 1
            setosa_err = True
    for j in range(10):
        result = classify(data["versicolor"]["testing"][j], LDA_flag, force_diagonal)
        testing_tries += 1
        if result != "versicolor":
            testing_wrong += 1
            versicolor_err = True
    for k in range(10):
        result = classify(data["virginica"]["testing"][k], LDA_flag, force_diagonal)
        testing_tries += 1
        if result != "virginica":
            testing_wrong += 1
            virgnica_err = True

    testing_error_rate = testing_wrong/testing_tries
    print "TESTING ERROR RATE=", testing_error_rate * 100, "%"
    # Test linear separability
    if LDA_flag:
        print "Class Iris-Setosa is " + ("NOT " if setosa_err else "") + "linearly separable"
        print "Class Iris-Versicolor is " + ("NOT " if versicolor_err else "") + "linearly separable"
        print "Class Iris-virginica is " + ("NOT " if virginica_err else "") + "linearly separable"


if __name__ == "__main__":
    num_feat = 3
    data = process_data_set(use_cols=(0, 1, 2, 3))

    assert(len(data["setosa"]["training"]) == len(data["versicolor"]["training"]) == len(data["virginica"]["training"]))
    assert(len(data["setosa"]["testing"]) == len(data["versicolor"]["testing"]) == len(data["virginica"]["testing"]) == 10)

    ## TRAINING
    mu1, sigma1 = train(data["setosa"]["training"])
    mu2, sigma2 = train(data["versicolor"]["training"])
    mu3, sigma3 = train(data["virginica"]["training"])

    ## TESTING
    evaluate_model(LDA_flag=False) # QDA
    evaluate_model(LDA_flag=True) # LDA
    # Force covariance matrix to a diagonal matrix
    # evaluate_model(LDA_flag=False, force_diagonal=True) # QDA
    # evaluate_model(LDA_flag=True, force_diagonal=True) # LDA
