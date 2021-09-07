"""
Registration module main code.
"""

import numpy as np
from scipy import ndimage
import registration_util as util
import matplotlib.pyplot as plt
import registration as reg
from IPython.display import display, clear_output
import hist_boost as fst

# SECTION 1. Geometrical transformations


def identity():
    # 2D identity matrix.
    # Output:
    # T - transformation matrix

    T = np.eye(2)

    return T


def scale(sx, sy):
    # 2D scaling matrix.
    # Input:
    # sx, sy - scaling parameters
    # Output:
    # T - transformation matrix
    T = np.array([[sx,0],[0,sy]])

    return T


def rotate(phi):
    # 2D rotation matrix.
    # Input:
    # phi - rotation angle
    # Output:
    # T - transformation matrix
    list = [np.cos(phi),-np.sin(phi),np.sin(phi),np.cos(phi)]
    T = np.reshape(np.array(list),[2,2])
    return T



def shear(cx, cy):
    # 2D shearing matrix.
    # Input:
    # cx - horizontal shear
    # cy - vertical shear
    # Output:
    # T - transformation matrix
    list = [1,cx,cy,1]
    T = np.reshape(np.array(list),[2,2])
    return T


def reflect(rx, ry):
    # 2D reflection matrix.
    # Input:
    # rx - horizontal reflection (must have value of -1 or 1)
    # ry - vertical reflection (must have value of -1 or 1)
    # Output:
    # T - transformation matrix

    allowed = [-1, 1]
    if rx not in allowed or ry not in allowed:
        T = 'Invalid input parameter'
        return T
    else:
    
        list = [rx,0,0,ry]
        T = np.reshape(np.array(list),[2,2])
    return T


# SECTION 2. Image transformation and least squares fitting

def image_transform(I, Th,  output_shape=None):
    # Image transformation by inverse mapping.
    # Input:
    # I - image to be transformed
    # Th - homogeneous transformation matrix
    # output_shape - size of the output image (default is same size as input)
    # Output:
    # It - transformed image
	# Xt - remapped coordinates
    # we want double precision for the interpolation, but we want the
    # output to have the same data type as the input - so, we will
    # convert to double and remember the original input type

    input_type = type(I);

    # default output size is same as input
    if output_shape is None:
        output_shape = I.shape

    # spatial coordinates of the transformed image
    x = np.arange(0, output_shape[1])
    y = np.arange(0, output_shape[0])
    xx, yy = np.meshgrid(x, y)

    # convert to a 2-by-p matrix (p is the number of pixels)
    X = np.concatenate((xx.reshape((1, xx.size)), yy.reshape((1, yy.size))))
    # convert to homogeneous coordinates
    Xh = util.c2h(X)

    #------------------------------------------------------------------#
    Tinv = np.linalg.inv(Th)
    Xt = Tinv.dot(Xh)
    
    #------------------------------------------------------------------#

    It = ndimage.map_coordinates(I, [Xt[1,:], Xt[0,:]], order=1, mode='constant').reshape(I.shape)

    return It, Xt


def ls_solve(A, b):
    # Least-squares solution to a linear system of equations.
    # Input:
    # A - matrix of known coefficients
    # b - vector of known constant term
    # Output:
    # w - least-squares solution to the system of equations
    # E - squared error for the optimal solution
    #------------------------------------------------------------------#
    transA = np.transpose(A)
    AA = (transA).dot(A)
    invAA = np.linalg.inv(AA)
    new = invAA.dot(transA)
    wfirst = new.dot(b)
    w = np.asarray(wfirst)
    #------------------------------------------------------------------#

    # compute the error
    E = np.transpose(A.dot(w) - b).dot(A.dot(w) - b)

    return w, E


def ls_affine(X, Xm,returnerror=False):
    # Least-squares fitting of an affine transformation.
    # Input:
    # X - Points in the fixed image
    # Xm - Corresponding points in the moving image
    # Output:
    # T - affine transformation in homogeneous form.

    A = np.transpose(Xm)
    k = np.ones((A.shape[0],1))
    A = np.concatenate((A,k),axis=1)
    
    #------------------------------------------------------------------#
    xc = X[0,:]
    yc = X[1,:]
    wone,E1= (ls_solve(A,xc))
    wtwo,E2 = (ls_solve(A,yc))
    eye = np.identity(3)
    eye [0,:] = np.transpose(wone)
    eye [1,:] = np.transpose(wtwo)
    T = eye
    Error = np.sqrt(E1**2+E2**2)
    #------------------------------------------------------------------#
    if returnerror:
        return T,Error
    else:
        return T


# SECTION 3. Image simmilarity metrics


def correlation(I, J):
    # Compute the normalized cross-correlation between two images.
    # Input:
    # I, J - input images
    # Output:
    # CC - normalized cross-correlation
    # it's always good to do some parameter checks

    if I.shape != J.shape:
        raise AssertionError("The inputs must be the same size.")

    u = I.reshape((I.shape[0]*I.shape[1],1))
    v = J.reshape((J.shape[0]*J.shape[1],1))

    # subtract the mean
    u = u - u.mean(keepdims=True)
    v = v - v.mean(keepdims=True)

    #------------------------------------------------------------------#
    transu = np.transpose(u)
    length = np.linalg.norm(u,ord=2)*np.linalg.norm(v,ord=2)
    CC = transu.dot(v)/length
    #------------------------------------------------------------------#

    return CC


def joint_histogram(I, J, num_bins=16, minmax_range=None):
    # Compute the joint histogram of two signals.
    # Input:
    # I, J - input images
    # num_bins: number of bins of the joint histogram (default: 16)
    # range - range of the values of the signals (defaul: min and max
    # of the inputs)
    # Output:
    # p - joint histogram

    if I.shape != J.shape:
        raise AssertionError("The inputs must be the same size.")

    # make sure the inputs are column-vectors of type double (highest
    # precision)
    I = I.reshape((I.shape[0]*I.shape[1],1)).astype(float)
    J = J.reshape((J.shape[0]*J.shape[1],1)).astype(float)

    # if the range is not specified use the min and max values of the
    # inputs
    if minmax_range is None:
        minmax_range = np.array([min(min(I),min(J)), max(max(I),max(J))])

    # this will normalize the inputs to the [0 1] range
    I = (I-minmax_range[0]) / (minmax_range[1]-minmax_range[0])
    J = (J-minmax_range[0]) / (minmax_range[1]-minmax_range[0])

    # and this will make them integers in the [0 (num_bins-1)] range
    I = np.round(I*(num_bins-1)).astype(int)
    J = np.round(J*(num_bins-1)).astype(int)

    n = I.shape[0]
    hist_size = np.array([num_bins,num_bins])

    # initialize the joint histogram to all zeros
    p = np.zeros(hist_size)

    for k in range(n):
        p[I[k], J[k]] = p[I[k], J[k]] + 1

    #------------------------------------------------------------------#
    p = np.divide(p, n)
    #------------------------------------------------------------------#
    
    return p


def mutual_information(p):
    # Compute the mutual information from a joint histogram.
    # Input:
    # p - joint histogram
    # Output:
    # MI - mutual information in nat units
    # a very small positive number

    EPSILON = 10e-10

    # add a small positive number to the joint histogram to avoid
    # numerical problems (such as division by zero)
    p += EPSILON

    # we can compute the marginal histograms from the joint histogram
    p_I = np.sum(p, axis=1)
    p_I = p_I.reshape(-1, 1)
    p_J = np.sum(p, axis=0)
    p_J = p_J.reshape(1, -1)

    #------------------------------------------------------------------#
    MI = np.sum(np.multiply( p,np.log(np.divide(p,p_I.dot(p_J)))))
    #------------------------------------------------------------------#

    return MI


def mutual_information_e(p):
    # Compute the mutual information from a joint histogram.
    # Alternative implementation via computation of entropy.
    # Input:
    # p - joint histogram
    # Output:
    # MI - mutual information in nat units
    # a very small positive number

    EPSILON = 10e-10

    # add a small positive number to the joint histogram to avoid
    # numerical problems (such as division by zero)
    p += EPSILON

    # we can compute the marginal histograms from the joint histogram
    p_I = np.sum(p, axis=1)
    p_I = p_I.reshape(-1, 1)
    p_J = np.sum(p, axis=0)
    p_J = p_J.reshape(1, -1)

    #------------------------------------------------------------------#
    H_I = -1*np.sum(np.multiply(p_I,np.log(p_I)))
    H_J = -1*np.sum(np.multiply(p_J,np.log(p_J)))
    H_IJ = -1*np.sum(np.multiply(p,np.log(p)))

    MI = H_I + H_J - H_IJ
    #------------------------------------------------------------------#

    return MI


# SECTION 4. Towards intensity-based image registration


def ngradient(fun, x, h=1e-3):
    # Computes the derivative of a function with numerical differentiation.
    # Input:
    # fun - function for which the gradient is computed
    # x - vector of parameter values at which to compute the gradient
    # h - a small positive number used in the finite difference formula
    # Output:
    # g - vector of partial derivatives (gradient) of fun

    #------------------------------------------------------------------#
    # TODO: Implement the  computation of the partial derivatives of
    # the function at x with numerical differentiation.
    # g[k] should store the partial derivative w.r.t. the k-th parameter

    sizing = x.size
    g = np.zeros((sizing,1))

    for i in range(sizing):
        p = np.zeros((sizing,1))
        p[i,0] = h/2

        localmax = x+p
        localmin = x-p

        g[i] = (fun(localmax)-fun(localmin))/h
    g = g.reshape(sizing,1) 
    return g


def rigid_corr(I, Im, x, return_transform=True):
    # Computes normalized cross-correlation between a fixed and
    # a moving image transformed with a rigid transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the rigid transform: the first element
    #     is the rotation angle and the remaining two elements
    #     are the translation
    # return_transform: Flag for controlling the return values
    # Output:
    # C - normalized cross-correlation between I and T(Im)
    # Im_t - transformed moving image T(Im)
    # Th - transformation matrix (only returned if return_transform=True)

    SCALING = 100

    # the first element is the rotation angle
    T = rotate(x[0])


    # the remaining two element are the translation
    #
    # the gradient ascent/descent method work best when all parameters
    # of the function have approximately the same range of values
    # this is  not the case for the parameters of rigid registration
    # where the transformation matrix usually takes  much smaller
    # values compared to the translation vector this is why we pass a
    # scaled down version of the translation vector to this function
    # and then scale it up when computing the transformation matrix
    Th = util.t2h(T, x[1:]*SCALING)

    # transform the moving image
    Im_t, Xt = image_transform(Im, Th)

    # compute the similarity between the fixed and transformed
    # moving image
    C = correlation(I, Im_t)

    if return_transform:
        return C, Im_t, Th
    else:
        return C


def affine_corr(I, Im, x, return_transform=True):
    # Computes normalized cross-corrleation between a fixed and
    # a moving image transformed with an affine transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the rigid transform: the first element
    #     is the roation angle, the second and third are the
    #     scaling parameters, the fourth and fifth are the
    #     shearing parameters and the remaining two elements
    #     are the translation
    # return_transform: Flag for controlling the return values
    # Output:
    # C - normalized cross-correlation between I and T(Im)
    # Im_t - transformed moving image T(Im)
    # Th - transformation matrix (only returned if return_transform=True)
    
    NUM_BINS = 64
    SCALING = 100

    #------------------------------------------------------------------#
    T = rotate(x.item(0)).dot(scale(x.item(1),x.item(2))).dot(shear(x.item(3),x.item(4)))
    Th = util.t2h(T,x[5:]*SCALING)
    Im_t, Xt = image_transform(Im, Th)

    C = correlation(I,Im_t)
    #------------------------------------------------------------------#

    if return_transform:
        return C, Im_t, Th
    else:
        return C


def affine_mi(I, Im, x, return_transform=True, fast=False):
    # Computes mutual information between a fixed and
    # a moving image transformed with an affine transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the rigid transform: the first element
    #     is the rotation angle, the second and third are the
    #     scaling parameters, the fourth and fifth are the
    #     shearing parameters and the remaining two elements
    #     are the translation
    # return_transform: Flag for controlling the return values
    # Output:
    # C - normalized cross-correlation between I and T(Im)
    # Im_t - transformed moving image T(Im)
    # Th - transformation matrix (only returned if return_transform=True)

    NUM_BINS = 64
    SCALING = 100
    
    T = rotate(x[0]).dot(scale(x[1],x[2])).dot(shear(x[3],x[4]))
    Th = util.t2h(T, x[5:]*SCALING)
    
    Im_t, Xt = image_transform(Im, Th)
    C = correlation(I, Im_t)
    
    if fast:
        p = fst.hist_2d_boosted(I,Im_t)
    else:
        p = joint_histogram(I,Im_t)

    MI= mutual_information(p)
    

    if return_transform:
        return MI, Im_t, Th
    else:
        return MI
    #------------------------------------------------------------------#
