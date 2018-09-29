"""
Created on Wed Sep 18 11:27:59 2018

@author: anujgrgv
"""
# F18 EECS 442 HW1p1 Linear Least Square

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def _fit_curve(x0,y0):
    '''implement linear least square to fit the data with the cubic curve

    Args:
        x0: the vector containing the x coordinates of data points
        y0: the vector containing the y coordinates of data points
    Returns:
        The vector C containing the coefficients of the cubic curve
        Y=C(0)*X.^n + C(1)*X.^(n-1) +...+ C(n-1)*X + C(n)
    '''
    #------------------------------------------------
    # FILL YOUR CODE HERE
    #
    #------------------------------------------------

def _viz_curve(C,x0,y0):
    '''Plot the original data and the curve that fits the data

    Args:
        x0: the vector containing the x coordinates of data points
        y0: the vector containing the y coordinates of data points
        C : the vector containing the coefficients of the cubic curve
    '''
    #------------------------------------------------
    # FILL YOUR CODE HERE
    #
    #------------------------------------------------

def run():
    #loading data
    data = np.load("hw1_p1.npy")
    x = data.item().get('X')
    y = data.item().get('Y')

    #getting coefficients
    Coeff = _fit_curve(x,y)
    print(Coeff)

    #plotting the curve
    _viz_curve(Coeff,x,y)


if __name__ == "__main__":
    run()
