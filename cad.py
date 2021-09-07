"""
CAD module code main code.
"""

import numpy as np
import cad_util as util


def sigmoid(a):
    # Computes the sigmoid function
    # Input:
    # a - value for which the sigmoid function should be computed
    # Output:
    # s - output of the sigmoid function

    # s = 1/ (1+e^-x)
    s = np.divide(1.0, 1.0+np.exp(-1*a))

    return s


def lr_nll(X, Y, Theta):
    # Computes the negative log-likelihood (NLL) loss for the logistic
    # regression classifier.
    # Input:
    # X - the data matrix
    # Y - targets vector
    # Theta - parameters of the logistic regression model
    # Ouput:
    # L - the negative log-likelihood loss

    # compute the predicted probability by the logistic regression model
    p = sigmoid(X.dot(Theta))
 
    L = -1*np.sum(np.multiply(Y,np.log(p))+np.multiply(1-Y,np.log(1-p)))
 
    return L


def lr_agrad(X, Y, Theta):
    # Gradient of the negative log-likelihood for a logistic regression
    # classifier.
    # Input:
    # X - the data matrix
    # Y - targets vector
    # Theta - parameters of the logistic regression model
    # Example inputs:
    # X - training_x_ones.shape=(100, 1729)
    # Y - training_y[idx].shape=(100, 1)
    # Theta - Theta.shape=(1729, 1)
    # Ouput:
    # g - gradient of the negative log-likelihood loss
    
    a = X.dot(Theta)
    p = sigmoid(a)
    g = np.sum((p - Y)*X, axis=0).reshape(1,-1)

    return g

def ngradient(fun, x, h=1e-3):
    # Computes the derivative of a function with numerical differentiation.
    # Input:
    # fun - function for which the gradient is computed
    # x - vector of parameter values at which to compute the gradient
    # h - a small positive number used in the finite difference formula
    # Output:
    # g - vector of partial derivatives (gradient) of fun

    sizing = x.size
    g      = np.zeros((sizing,1))

    for i in range(sizing):
        p = np.zeros((sizing,1))
        p[i,0] = h/2

        localmax = x+p
        localmin = x-p

        g[i] = (fun(localmax)-fun(localmin))/h
    g = g.reshape(sizing,1) 
    return g

def ls_solve(A, b):
    # Least-squares solution to a linear system of equations.
    # Input:
    # A - matrix of known coefficients
    # b - vector of known constant term
    # Output:
    # w - least-squares solution to the system of equations

    transA = np.transpose(A)
    AA     = (transA).dot(A)
    invAA  = np.linalg.inv(AA)
    new    = invAA.dot(transA)

    wfirst = new.dot(b)
    w      = np.asarray(wfirst)


    # compute the error
    E = np.transpose(A.dot(w) - b).dot(A.dot(w) - b)

    return w, E

def addones(X):
    # Add a column of all ones to a data matrix.
    # Input/output:
    # X - the data matrix

    if len(X.shape) < 2:
        X = X.reshape(-1,1)

    r, c = X.shape
    one_vec = np.ones((r,1))
    X = np.concatenate((X,one_vec), axis=1)
    return X

def montageRGB(X, ax):
    # Creates a 2D RGB montage of image slices from a 4D matrix
    # Input:
    # X - 4D matrix containing multiple 2D image slices in RGB format
    #     to be displayed as a montage / mosaic
    #
    # Adapted from http://www.datawrangling.org/python-montage-code-for-displaying-arrays/

    m, n, RGBval, count = X.shape
    mm = int(np.ceil(np.sqrt(count)))
    nn = mm
    M = np.zeros((mm * m, nn * n, 3))
    image_id = 0

    for j in np.arange(mm):
        for k in np.arange(nn):
            if image_id >= count:
                break
            sliceM, sliceN = j * m, k * n
            M[sliceN:sliceN + n, sliceM:sliceM + m, :] = X[:, :, :, image_id]
            image_id += 1

    M = np.flipud(np.rot90(M)).astype(np.uint8)
    ax.imshow(M)

    return M
