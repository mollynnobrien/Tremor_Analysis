import cv2
import numpy as np
import sys
import math
import os
from numpy.linalg import inv
import random
import re
import pickle

#_________________________________________________________________
# 
# Function: loadCamCal
# 		load camera calibration results 
# Author: Molly O'Brien
# Input: filename 
# Return: foundL, rvecsL, tvecsL, foundR, rvecsR, tvecsR
#-----------------------------------------------------------------
def loadCamCal(filename):
	# Read in calibration parameters        
	with open(filename, 'rb') as f:
            calResults = pickle.load(f)

	foundL = calResults[0]
	rvecsL = calResults[1]
	tvecsL = calResults[2]
	foundR = calResults[3]
	rvecsR = calResults[4]
	tvecsR = calResults[5]
	image_list = calResults[6]

	return foundL, rvecsL, tvecsL, foundR, rvecsR, tvecsR, image_list
#_________________________________________________________________

#_________________________________________________________________
#
# Function: loadSterCal 
#		load stereo calibration results 
# Author: Molly O'Brien 
# Input: filename
# Return: retval, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F
#-----------------------------------------------------------------
def loadSterCal(filename):
	# Read in stereo calibration parameters 
	with open(filename, 'rb') as f:
	    sterResults = pickle.load(f)

    # extract the different values from list 
	retval = sterResults[0]
	cameraMatrixL = sterResults[1]
	distCoeffsL = sterResults[2]
	cameraMatrixR = sterResults[3]
	distCoeffsR = sterResults[4]
	R = sterResults[5]
	T = sterResults[6]
	E = sterResults[7]
	F = sterResults[8]

	return retval, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F
#_________________________________________________________________

