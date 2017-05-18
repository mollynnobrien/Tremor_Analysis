import cv2
import numpy as np
import sys
import math
import os
from numpy.linalg import inv
import random
import re
import pickle

#___________________________________________________________________
#
# Function: loadCheckerBoard
# 	    	Create calibration points for checkerboard in "world" coordinates
#
# Author:   	Molly O'Brien
# Input: 	num: which checkerboard being used
# 			num_ims: number of calibration images
# Return:	objectPoints: the 3D coordinates of internal checkerboard corners in world
#			cooridnates 
#------------------------------------------------------------------- 
def loadCheckerBoard(num, num_ims):
	# -----------------checkerboard 1 info---------------- 

	if(num == 0):
		# num internal corners on each column: 
		col_corn = 6
		# num internal corners on each row:
		row_corn = 6

		# number of internal corners to be detected	
		numPoints = col_corn*row_corn
		# the corner positions in each frame
		objectPoints1Frame = np.zeros((numPoints, 3), np.float32)
		
		# vector with corner positions corresponding to each input image
		objectPoints = []
		# counter to keep track of which point we are on
		pt = 0
		# square side length is 6 mm
		side_len = 6
		# iterate over each internal corner point
		for row_pt in range(row_corn):
			for col_pt in range(col_corn): 
				# coordinates on checkerboard in mm, z always 0 bc on a plane
				objectPoints1Frame[pt] = [6*col_pt, 6*row_pt, 0]	
		 		pt = pt + 1
		# create ref corner positions for each input image
		for im in range(num_ims):
			objectPoints.append(objectPoints1Frame)

	if(num == 2):
		# num internal corners on each column: 
		col_corn = 10
		# num internal corners on each row:
		row_corn = 11

		# number of internal corners to be detected	
		numPoints = col_corn*row_corn
		# the corner positions in each frame
		objectPoints1Frame = np.zeros((numPoints, 3), np.float32)
		
		# vector with corner positions corresponding to each input image
		objectPoints = []
		# counter to keep track of which point we are on
		pt = 0
		# square side length is 2.137 measured 4/14/17 w digital caliper 
		side_len = 2.137
		# iterate over each internal corner point
		for col_pt in range(col_corn):
			for row_pt in range(row_corn): 
				# Start from right side of checkerboard 
				COL_PT = col_corn - col_pt
				# coordinates on checkerboard in mm, z always 0 bc on a plane
				objectPoints1Frame[pt] = [side_len*COL_PT, side_len*row_pt, 0]	
		 		pt = pt + 1
		# create ref corner positions for each input image
		for im in range(num_ims):
			objectPoints.append(objectPoints1Frame)

	if(num == 3):
		# num internal corners on each column: 
		col_corn = 4
		# num internal corners on each row:
		row_corn = 4

		# number of internal corners to be detected	
		numPoints = col_corn*row_corn
		# the corner positions in each frame
		objectPoints1Frame = np.zeros((numPoints, 3), np.float32)
		
		# vector with corner positions corresponding to each input image
		objectPoints = []
		# counter to keep track of which point we are on
		pt = 0
		# square side length is 6 mm
		side_len = 2
		# iterate over each internal corner point
		for row_pt in range(row_corn):
			for col_pt in range(col_corn): 
				# Start from right side of checkerboard 
				#ROW_PT = row_corn - row_pt
				# coordinates on checkerboard in mm, z always 0 bc on a plane
				objectPoints1Frame[pt] = [side_len*col_pt, side_len*row_pt, 0]	
		 		pt = pt + 1
		# create ref corner positions for each input image
		for im in range(num_ims):
			objectPoints.append(objectPoints1Frame)

	#---------------------------------------------------------
	return objectPoints, col_corn, row_corn
#_______________________________________________________________________
