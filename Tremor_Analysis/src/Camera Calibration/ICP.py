import cv2
import numpy as np
import sys
import math
import os
from numpy.linalg import inv
import random
import re
import pickle
from dist import dist
from scipy.special import factorial
import pdb

sys.path.insert(0, 'src/Color Marker Detection/')
sys.path.insert(0, 'src/Bundle Adjustment/')
sys.path.insert(0, 'src/Camera Calibration/')
sys.path.insert(0, 'src/Frequency Analysis/')

# import custom functions
from loadParams import loadCamCal, loadSterCal
from blobMatch import blobMatch

#___________________________________________________________________
#
# Function: findTransform 
# 			Find the rotation between 2 point clouds point correspondences  
# Inputs: 	A: point cloud 1 (points ordered so that pt 1 in A matches
#			pt 1 in B)
# 			B: point cloud 2 
# Return: 	R: rotation between A and B
#			t: translation between A and B
#					R*A + t = B
# Uses method from CIS 1 HW and notes
#-------------------------------------------------------------------
def findTransform(A, B):

	#--------------------------------------------------
	# Prune matches that have 3D distance > 10mm
	Aprune = []
	Bprune = []

	#--------------------------------------------------
	# Find t: 
	meanA = np.mean(A, axis=0)
	meanB = np.mean(B, axis=0)

	# t = mean displacement
	t = meanB - meanA
	# subtract off mean from point clouds
	A = A - meanA
	B = B - meanB

	
	#---------------------------------------------------
	# Find R: 
	# use matrix H to find R
	H = np.zeros((3,3))

	for a, b in zip(A,B):
	#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
			# form matrix H: 
			H[0,0] = H[0,0] + a[0]*b[0]
			H[0,1] = H[0,1] + a[0]*b[1]
			H[0,2] = H[0,2] + a[0]*b[2]

			H[1,0] = H[1,0] + a[1]*b[0]
			H[1,1] = H[1,1] + a[1]*b[1]
			H[1,2] = H[1,2] + a[1]*b[2]

			H[2,0] = H[2,0] + a[2]*b[0]
			H[2,1] = H[2,1] + a[2]*b[1]
			H[2,2] = H[2,2] + a[2]*b[2]

		
	#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	
	delta = np.array([H[1,2]-H[2,1], H[2,0]-H[0,2], H[0,1]-H[1,0]])
	delta.shape = (1,3)
	
	# form G
	G = np.zeros((4,4))
	G[0,0] = np.trace(H)
	G[0,1:4] = delta
	# rows 2-4
	delta.shape = (3,1)
	G[1:4, 0:1] = delta
	G[1:4, 1:4] = H+H.transpose()-np.eye((3))*np.trace(H)
	
	# eigen decomposition of G
	eigVals, eigVects = np.linalg.eig(G)

	# find the largest eigenvalue
	eigIndex = eigVals.argmax()
	# get the vector corresponding to largest eigenvector
	rotVect = eigVects[:,eigIndex]	
	# the eigenvector w largest eigenvalue is the rotation quaternion
	R = quatToRotM(rotVect)

	# form rotation matrix from quaternion
	T = np.zeros((4,4))
	T[0:3, 0:3] = R
	t.shape = (3,1)
	T[0:3, 3:4] = t
	T[3:4, 3:4] = 1
	
	return T
	
#___________________________________________________________________

#___________________________________________________________________
#
# quatToRotM: turns quaternion into rotation matrix, assumed quaternion: 
# 				q = [w x y z] where w is the scalar, and x y z is the vector
#
# \params[in] q: quaternion with the scalar value first [qw, qx, qy, qz]
# \return R: 3x3 rigid rotation matrix
#
# Reference: used the conversion method from 
# 	http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
#____________________________________________________________________

def quatToRotM(q):
	# extract the components from the quaternion
	qw = q[0]
	qx = q[1]
	qy = q[2]
	qz = q[3]

	# declare R 
	R = np.zeros((3,3))

	# fill R
	# From Dr. Taylor's CIS 1 lecture notes (Rigid 3D to 3D pg 22)
	R[0,0] = qw*qw + qx*qx - qy*qy - qz*qz
	R[0,1] = 2*qx*qy - 2*qz*qw
	R[0,2] = 2*qx*qz + 2*qy*qw

	R[1,0] = 2*qx*qy + 2*qz*qw
	R[1,1] = qw*qw - qx*qx + qy*qy - qz*qz 
	R[1,2] = 2*qy*qz - 2*qx*qw

	R[2,0] = 2*qx*qz - 2*qy*qw 
	R[2,1] = 2*qy*qz + 2*qx*qw
	R[2,2] = qw*qw - qx*qx - qy*qy + qz*qz

	return R

#____________________________________________________________________

#____________________________________________________________________
#
# Function: stabilizePoints
#
# 			Read in 3D point clouds from video frame and frame-to-frame transforms. 
#			Apply frame-to-frame transform iteratively to transform each point
#			cloud into the first frame's coordinates
#
# \params[in] points3D: list of 3D point cloud found in each frame 
# \params[in] T: 	T[i] is the transform from frame i's cooridnate to frame (i-1)'s
#					coordinates. T[0]*T[1]*...T[i]*points3D[i] transforms 
#					points3D[i] into the first frame's coordinates
# \return stabilizedPoints: 	points3D from all frames transformed into first frame's coordinates
#---------------------------------------------------------------------
def stabilizePoints(points3D, T, outputFile):
	# first transform is identity
	Tprev = np.eye((4))
	# frame counter
	frameCount = 1
	# declare list for stabilized points
	stabilizedPoints = []
	# iterate through each frame
	for framePoints, frameT in zip(points3D, T):
		outputFile.write("\n ---------------------------------- \n")
		outputFile.write("Frame Number: " + str(frameCount) + '\n')
		outputFile.write("delta T: " + str(frameT) + '\n')

		# multiply T[0,i-1]*T[i] 
		Tcur = np.dot(Tprev, frameT)
		
		outputFile.write("T from 1 to " + str(frameCount) + ": " + str(Tcur) +'\n')
		# extract rotation 
		R = Tcur[0:3, 0:3]
		# extract translation
		t = Tcur[0:3, 3:4]
		# rotate points
		stableFrame = rotatePoints(framePoints, R, t)

		# get variables ready for next iteration
		Tprev = Tcur
		# save stabilized points
		stabilizedPoints.append(stableFrame)
		# increment frame counter
		frameCount = frameCount + 1
		
	# return the stabilized points
	return stabilizedPoints
#____________________________________________________________________

#___________________________________________________________________
# 
# Function: rotatePoints
# 
# 			given a list of 3D coordinates, transform by given 4x4 homogeneous 
#			transform 
# 
# \params[in] points: 	Nx3 array of points 
# \params[in] R:			3x3 rotation matrix
# \params[in] t: 		3x1 translation vector 
# \return points_rot: 	points transformed by T
#--------------------------------------------------------------------
def rotatePoints(points, R, t):
	#print("in rotate points")
	pt = 0
	worldPoints = np.zeros((3,len(points)))
	for point in points:
		print("point: " + str(point))
		# make point a column vector
		point.shape = (3,1)
		# [R_world_cam|t]*corners_world
		worldPoint = np.dot(R, point) + t
		# make cameraPoint a column vector
		worldPoint.shape = (3,1)
		# save cameraPoint in matrix cameraPoints
		worldPoints[0:3,pt:pt+1] = worldPoint
		# increment pt
		pt = pt + 1

	worldPoints = worldPoints.transpose()
	
	return worldPoints

#___________________________________________________________________
#
