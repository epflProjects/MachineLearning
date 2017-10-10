# -*- coding: utf-8 -*-
# Authors : Matteo Giorla, Bastian Nanchen, Arnaud Pannatier
# EPFL Fall 2017
# Machine Learning Course
# Project : 1
""" Testing of the functions of Machine Learning Project 1."""
import numpy as np
from implementations import *
import numpy.matlib
import datetime

def construct_perfect_linear_data(a,b):
	x = np.arange(1e5)
	x, mean_x, std_x = standardize(x)
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

def test_linear_regression():
	a = 10
	b = 20

	(tx,y) = construct_perfect_linear_data(a,b)

	initial_w = np.array([9.0,20.0])
	max_iters = 50
	gamma = 0.8
	lambda_ = 0

	# Test 1 : Linear Regression using Gradient Descent 
	print("\n")
	print(" ------------ Test 1 : Linear Regression using Gradient Descent ------------ ")  
	
	start_time = datetime.datetime.now()
	(w, loss) = least_squares_GD(y, tx, initial_w, max_iters, gamma)
	end_time = datetime.datetime.now()
	exection_time = (end_time - start_time).total_seconds()
	print("Weights :", w, "Loss : ", loss)

	assert (loss < 1e-5), "Differents weights !"
	print("Linear Regression using Gradient Descent is working. Execution time={t:.3f} seconds".format(t=exection_time)) 
	print("\n")

	# Test 2 : Linear Regression using Gradient Descent 
	print(" ------------ Test 2 : Linear Regression using Stochastic Gradient Descent ------------ ")  
	
	start_time = datetime.datetime.now()
	(w, loss) = least_squares_SGD(y, tx, initial_w, max_iters, gamma)
	end_time = datetime.datetime.now()
	exection_time = (end_time - start_time).total_seconds()
	print("Weights :", w, "Loss : ", loss)

	assert (loss < 1e-2), "Differents weights !"
	print("Linear Regression using Stochastic Gradient Descent is working. Execution time={t:.3f} seconds".format(t=exection_time)) 
	print("\n") 
	
	# Test 3 : Least Squares 
	print(" ------------ Test 3 : Least Squares regression ------------ ")  

	start_time = datetime.datetime.now()
	(w, loss) = least_squares(y,tx)
	end_time = datetime.datetime.now()
	exection_time = (end_time - start_time).total_seconds()
	print("Weights :", w, "Loss : ", loss)

	assert (loss < 1e-5), "Differents weights !"
	print("Least Squares regression is working. Execution time={t:.3f} seconds".format(t=exection_time)) 
	print("\n") 

	# Test 4 : Ridge regression 
	print(" ------------ Test 4 : Ridge regression ------------ ")  

	start_time = datetime.datetime.now()
	(w, loss) = ridge_regression(y, tx, lambda_)
	end_time = datetime.datetime.now()
	exection_time = (end_time - start_time).total_seconds()
	print("Weights :", w, "Loss : ", loss)

	assert (loss < 1e-5), "Differents weights !"
	print("Ridge regression is working. Execution time={t:.3f} seconds".format(t=exection_time)) 
	print("\n") 

test_linear_regression()
