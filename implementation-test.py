# -*- coding: utf-8 -*-
# Authors : Matteo Giorla, Bastian Nanchen, Arnaud Pannatier
# EPFL Fall 2017
# Machine Learning Course
# Project : 1
""" Testing of the functions of Machine Learning Project 1."""
import numpy as np
from implementations import *
import numpy.matlib

def construct_perfect_linear_data(a,b):
	x = np.arange(10000)
	y = a+b*x
	tx = build_poly(x,1)

	return (tx,y)

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    degree_range = np.arange(0,degree+1)
    degree_array = np.matlib.repmat(degree_range, len(x), 1)
    x_array = np.matlib.repmat(x, degree+1,1).T
    
    return(np.power(x_array,degree_array)) 

def test_least_square():
	a = 10
	b = 20

	(tx,y) = construct_perfect_linear_data(a,b)
	(w, loss) = least_squares(y,tx)

	print((w, loss))

	assert (loss < 1e-10), "Differents weights !"
	print("Least Squares regression is working") 

test_least_square()
