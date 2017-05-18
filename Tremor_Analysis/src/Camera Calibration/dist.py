import cv2
import numpy as np
import sys
import math
import os
from numpy.linalg import inv
import random
import re
import pickle

#_______________________________________________________________________
#
# Function: dist
#		find the Euclidean distance between two points
# Author: Molly O'Brien
# Inputs: x: point 1
#	  y: point 2
# Return: D: distance
#-----------------------------------------------------------------------
def dist(x,y):
	# find the lengths of the vectors
	Lx = len(x)
	Ly = len(y)

	# return None if not same length
	if(Lx != Ly): 
		print("Error in dist: length of x is not equal to length of y")
		return None 
	# add up square of different in each direction 
	squareD = 0 
	for entry in range(Lx):
		squareD = squareD + (x[entry]-y[entry])**2
	# take sqrt to find distance 
	D = math.sqrt(squareD)

	return D
#_______________________________________________________________________
