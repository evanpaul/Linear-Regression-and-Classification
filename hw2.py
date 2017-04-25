#!/usr/bin/python
import math
import numpy as np

# Process data into three matrices (split into training and testing) corresponding to classification
# REVIEW Something is weird with floats
def process_data_set():
    x1, x2, x3, x4 = np.genfromtxt("iris.data", delimiter=",", usecols=(
        0, 1, 2, 3), unpack=True, dtype=float)  # columns: x1, x2, x3, x4
    # Iris-setosa
    setosa_train_rows = []
    setosa_test_rows = []
    for i in range(50):
        row = [x1[i], x2[i], x3[i], x4[i]]
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
        row = [x1[j], x2[j], x3[j], x4[j]]
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
        row = [x1[k], x2[k], x3[k], x4[k]]
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
    mu = np.array([0.0, 0.0, 0.0, 0.0])
    for i in range(4):  # 4 = number of features
        col = x[:, i]  # ith column vector
        mu[i] = math.fsum(col) / len(col)

    # Estimate sigma (covariance) REVIEW
    sigma = np.zeros(shape=(4, 4))
    for j in range(len(x)):
        a = x[j] - mu
        b = np.transpose(x[j] + mu)
        outer_product = np.outer(a, b)  # outer product

        sigma += outer_product
    sigma = sigma / len(x)

    return mu, sigma
# Gaussian


def gaussian_pdf(x, mu, sigma):
    P = 4  # dimensionality of x i.e. number of features
    # Just to be explicit that we're dealing with column vectors in these
    # formulas
    x.shape = (4, 1)
    mu.shape = (4, 1)

    # TODO Decide whethr to include constant term or not
    #const_term = (np.linalg.det(sigma)**(-0.5)) * ((2 * math.pi)**(-P / 2))
    a = -0.5 * np.transpose(x - mu)
    b = np.asmatrix(a) * np.linalg.inv(sigma)
    c = b * np.asmatrix(x - mu)
    # prob = const_term * math.exp(c)
    prob = math.exp(c)
    return prob

# Classify input vector x into one of the three classes using Quadratic Discriminant Analysis (QDA) by default
# LDA=True to use Linear Discriminant Analysis instead
def classify(x, LDA_flag):
    if(LDA_flag):
        avg_sigma = (sigma1 + sigma2 + sigma3)/3
        c1 = gaussian_pdf(x, mu1, avg_sigma)
        c2 = gaussian_pdf(x, mu2, avg_sigma)
        c3 = gaussian_pdf(x, mu3, avg_sigma)
    else:
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
    #print "x:", np.transpose(x), "is of class:", iris
    return iris

def evaluate_model(LDA_flag):
    if(LDA_flag):
        print "Classifying data using linear discriminant analysis"
    else:
        print "Classifying data using quadratic discriminant analysis"
    # Training data accuracy (less important)
    training_tries = 0.0
    training_wrong = 0.0
    # Any errors (in LDA) means the data is not linearly separable
    setosa_err = False
    versicolor_err = False
    virginica_err = False

    for i in range(40):
        result = classify(data["setosa"]["training"][i], LDA_flag)
        training_tries += 1
        if result != "setosa":
            training_wrong += 1
            setosa_err = True
            #print "True value: setosa, predicted:", result
    for j in range(40):
        result = classify(data["versicolor"]["training"][j], LDA_flag)
        training_tries += 1
        if result != "versicolor":
            training_wrong += 1
            versicolor_err = True
            #print "True value: versicolor, predicted:", result
    for k in range(40):
        result = classify(data["virginica"]["training"][k], LDA_flag)
        training_tries += 1
        if result != "virginica":
            training_wrong += 1
            virginica_err = True
            #print "True value: virginica, predicted:", result

    training_error_rate = training_wrong/training_tries
    print "TRAINING ERROR RATE=", training_error_rate * 100, "%"
    # Testing data accuracy (more important)
    testing_tries = 0.0
    testing_wrong = 0.0
    for i in range(10):
        result = classify(data["setosa"]["testing"][i], LDA_flag)
        testing_tries += 1
        if result != "setosa":
            testing_wrong += 1
            setosa_err = True
            #print "True value: setosa, predicted:", result
    for j in range(10):
        result = classify(data["versicolor"]["testing"][j], LDA_flag)
        testing_tries += 1
        if result != "versicolor":
            testing_wrong += 1
            versicolor_err = True
            #print "True value: versicolor, predicted:", result
    for k in range(10):
        result = classify(data["virginica"]["testing"][k], LDA_flag)
        testing_tries += 1
        if result != "virginica":
            testing_wrong += 1
            virgnica_err = True
            #print "True value: virginica, predicted:", result

    testing_error_rate = testing_wrong/testing_tries
    print "TESTING ERROR RATE=", testing_error_rate * 100, "%"
    # Test linear separability
    if LDA_flag:
        if setosa_err:
            print "Setosa is NOT linearly separable"
        else:
            print "Setosa is linearly separable"
        if versicolor_err:
            print "Versicolor is NOT linearly separable"
        else:
            print "Versicolor is linearly separable"
        if virginica_err:
            print "Virgnica is NOT linearly separable"
        else:
            print "Virgnica is linearly separable"


if __name__ == "__main__":
    data = process_data_set()
    # TRAINING
    mu1, sigma1 = train(data["setosa"]["training"])
    mu2, sigma2 = train(data["versicolor"]["training"])
    mu3, sigma3 = train(data["virginica"]["training"])
    # TESTING
    evaluate_model(False) # QDA
    evaluate_model(True) # LDA
