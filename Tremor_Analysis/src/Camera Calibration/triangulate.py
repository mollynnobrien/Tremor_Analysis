import cv2
import numpy as np
import sys
import math
import os
from numpy.linalg import inv
import random
import re
import pickle
import pdb

# import custom functions
from loadParams import loadCamCal, loadSterCal

#___________________________________________________________________
#
# Function: triangulate
# 	    	Given feature position in left and right image, find point 
#			position in 3D camera coordinates 
#
# Author:   	Molly O'Brien
# Input: 	camCalFile: camera calibration filename
#			sterCalFile: stereo calibration filename
#			pointsL: feature positions (in pixels) in left image
#			pointsR: feature positions (in pixels) in right image 
# Return:	worldPoints: feature points in 3D camera coordinates
#
#------------------------------------------------------------------- 
def triangulate(camCalFile, sterCalFile, pointsL, pointsR):
	# read in camera calibration parameters
	foundL, rvecsL, tvecsL, foundR, rvecsR, tvecsR, imageList = loadCamCal(camCalFile)
	# read in stereo calibration parameters
	retval, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F = loadSterCal(sterCalFile)	
	
	#----------------------------------------------------------
	# *********************************************************
	# From Austin's Code calibration.cpp :
	# Form projection matrices 
	# R from stereo calibration gives rotation from R to L camera
	R_right_left = R	
	# left extrinsics are just identity 
	extL = np.eye(3,4)
	# Transform from right to left cam coordinates 
	extR = np.zeros((3,4))
	# Add rotation matrix
	extR[0:3, 0:3] = R_right_left
	# T: offset from right to left camera coordinates, ensure a column vector
	T.shape = (3,1)
	# save T in extR
	extR[:,3:4] = T
	#---------------------------------------------------------------
	
	# intrinsics in projL(0:3,0:3)
	intrinL = cameraMatrixL
	# intrinsics in projR(0:3, 0:3)
	intrinR = cameraMatrixR

	# form projection matrices
	projL = np.dot(intrinL, extL)
	projR = np.dot(intrinR, extR)
	
	# make array worldPoints where triangulated points will go
	worldPoints = np.zeros((3, len(pointsL)))

	pt = 0
	# pdb.set_trace()
	for ptL, ptR in zip(pointsL, pointsR):
			# form matrix
			A = formA(ptL, projL, ptR, projR)
			# form vector
			b = formb(ptL, projL, ptR, projR)
			
			# result will be stored here
			x = np.zeros((3,1))
			x.shape = (3,1)
			# solve for point 
			found, x = cv2.solve(A, b, x, cv2.DECOMP_SVD)

			x.shape = (3,1)
			# save in array worldPoints 
			worldPoints[:, pt:pt+1] = x
			# increment counter 
			pt = pt + 1

	return worldPoints

#___________________________________________________________________

#___________________________________________________________________
#
# Function: formA
# 	    	Given points in left and right images and left and right 
# 			projection matrices, form matrix A to be used for triangulation 
#
# Author:  	Molly O'Brien, adapted from code by Austin Reiter
# Input: 	pointL: feature position (in pixels) in left image
#			projL: projection matrix for left image 
#			pointR: feature position (in pixels) in right image 
#			projR: projection matrix for right image
# Return:	A: [4x3] matrix, will be used to triangulate 3D point position
#
#------------------------------------------------------------------- 
def formA(pointL, projL, pointR, projR):
	# pdb.set_trace()
	# extract x,y from pointL
	xL = pointL[0]
	yL = pointL[1]
	# extract x,y from pointR
	xR = pointR[0]
	yR = pointR[1]

	A = np.zeros((4,3))

	for col in range(3):
		# two equations from left image
		
		A[0,col] = projL[0,col] - xL*projL[2,col]
		A[1,col] = projL[1,col] - yL*projL[2,col]
		# two equations from right image
		A[2,col] = projR[0,col] - xR*projR[2,col]
		A[3,col] = projR[1,col] - yR*projR[2,col]

	return A

#______________________________________________________________________

#___________________________________________________________________
#
# Function: formb
# 	    	Given points in left and right images and left and right 
# 			projection matrices, form matrix b to be used for triangulation 
#
# Author:  	Molly O'Brien, adapted from code by Austin Reiter
# Input: 	pointL: feature position (in pixels) in left image
#			projL: projection matrix for left image 
#			pointR: feature position (in pixels) in right image 
#			projR: projection matrix for right image
# Return:	b: [4x1] vector, will be used to triangulate 3D point position
#
#------------------------------------------------------------------- 
def formb(pointL, projL, pointR, projR):
	# extract x,y from pointL
	xL = pointL[0]
	yL = pointL[1]
	# extract x,y from pointR
	xR = pointR[0]
	yR = pointR[1]

	# declare b
	b = np.zeros((4,1))
	b.shape = (4,1)
	
	# form b
	b[0] = xL*projL[2,3]-projL[0,3]
	b[1] = yL*projL[2,3]-projL[1,3]
	b[2] = xR*projR[2,3]-projR[0,3]
	b[3] = yR*projR[2,3]-projR[1,3]


	return b

#______________________________________________________________________


