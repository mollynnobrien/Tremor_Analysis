#________________________________________________________
#
# 	Implement Bundle Adjustment using Rodrigues angles 
#________________________________________________________

import cv2
import numpy as np
import pdb
import scipy.optimize
import math
import numpy.linalg as LA
#_____________________________________________________
# 
# Function: apply_bundle_adjustment
#
#			given N frames of marker points and initial transform guesses
#			apply bundle adjustment to 80 frame batches to find optimal transforms 
#			for all frames to the initial frame
#
# \params[in] marker_points: 	list of array of background points for each frame
# \params[in] T:				list of array of 4x4 initial homogeneous transform 
#								from each frame to the first frame
# \return: T_bundle:			list of array of 4x4 optimal homogeneous transform 
#								from each frame to the first frame
#------------------------------------------------------
def apply_bundle_adjustment(marker_points, T):
	# total number of frames in data
	numFrames = len(T)
	# frames processing per batch of bundle adjustment
	framesPerBatch = 80
	# number of batches
	numBatches = int(np.floor(numFrames/framesPerBatch))
	# number of variables in optimization
	numOptVar = 6	
	# get first frame points
	points_0 = marker_points[0]
	
	print("num batches: " + str(numBatches))
	# do first batch 
	print("in batch 0 ")
	if(numFrames < framesPerBatch):
		observations = marker_points
		T_initial = T
		# do first batch
		P_optimal = bundleAdjustment(observations, T_initial)
		# get transforms 
		T_optimal = extractTransforms(P_optimal)
		# add these transforms to T
		T_bundle = T_optimal

	else:

		observations = marker_points[0:framesPerBatch]
		T_initial = T[0:framesPerBatch]
		# do first batch
		P_optimal = bundleAdjustment(observations, T_initial)
		# get transforms 
		T_optimal = extractTransforms(P_optimal)
		# add these transforms to T
		T_bundle = T_optimal

		for batch in range(1, numBatches):
			print("in batch "+str(batch))
			# frame indices
			idx_start = (batch)*framesPerBatch 
			idx_end = (batch+1)*framesPerBatch 
			
			# get this batches points and transforms 
			observations = [points_0]
			# initial transform is identity
			T_initial = [T[0]]

			# add rest of marker points
			observations.extend(marker_points[idx_start:idx_end])
			# add rest of transforms 
			T_initial.extend(T[idx_start:idx_end])

			# apply bundle adjustment 
			P_optimal = bundleAdjustment(observations, T_initial)
			# get transforms 
			T_optimal = extractTransforms(P_optimal)
			# add these transforms to T
			T_bundle.extend(T_optimal[1:framesPerBatch+1])
			

		# last round of bundle adjustmenent
		observations = marker_points[idx_end :numFrames]
		T_initial = T[idx_end:numFrames]
		# apply bundle adjustment 
		P_optimal = bundleAdjustment(observations, T_initial)
		# get transforms 
		T_optimal = extractTransforms(P_optimal)
		# add these transforms to T
		T_bundle.extend(T_optimal)

	# return the optimal transforms
	return T_bundle

#______________________________________________________

#______________________________________________________
#
# Function: bundleAdjustment
# 
#		Given a list of points in each frame and initial transforms for each frame to the first frame, find the optimal transform from each frame to the first frame. The goal is to stabilize background points in microscope video.
#
# \params[in] observations: 	list of array of background points for each frame
# \params[in] T_initial:		list of array of 4x4 initial homogeneous transform 
#								from each frame to the first frame
# \return P_optimal: 			1xn*6 vector with the optimal transforms for the n 
#								frames. Each 1x6 vector represents the 4x4
#								homogeneous transform from points in frame n to 
#								points in frame 0.
#------------------------------------------------------
def bundleAdjustment(observations, T_initial): 
	P_initial = extractVector(T_initial)
	# nonlinear optimization to find best P
	P_optimal, info, ier, msg = scipy.optimize.fsolve(errorFunction, P_initial, args=observations, full_output=True)

	# pdb.set_trace()

	return P_optimal

#_______________________________________________________
#
# Function: extractVector
#
# 			turn list of transforms into a list of vectors 
# 			[[T1][T2]...[TN]] 
#				--> 
#			[qw1 qx1 qy1 qz1 x1 y1 z1 .... qwN qxN qyN qzN xN yN zN]
#
# \params[in] T_list: list of N homogeneous transforms
# \return:	1xN*6 vector, each transform in T_list is converted 
#			into a 1x6 vector v_i = [rx, ry, rz, x, y, z] where rx, ry, and rz are 
#			the Rodrigues angles from the rotation in T_i
#----------------------------------------------------------
def extractVector(T_list):
	V = []
	# go through each transform 
	for T in T_list: 
		# rotation matrix to quaternion
		r, J = cv2.Rodrigues(T[0:3, 0:3])
		# extract the Rodrigues angles
		rx = r[0][0]
		ry = r[1][0]
		rz = r[2][0]
		# xyz offset
		x = T[0, 3]
		y = T[1, 3]
		z = T[2, 3]
		# extend vector
		V.extend([rx, ry, rz, x, y, z])

	return V

#________________________________________
#
# Function: errorFunction 
#
#			compute the error for a given vector of transforms P being optimized in bundle adjustment. 
#
# \param[in] P: 1xN*6 vector of parameters being optimized 
# \param[in] x: observations (3D points)
# \return error: 1xN*6 vector of the error associated with each transform parameter in P
#-------------------------------------------
def errorFunction(P, x):
	# P list of transforms 
	T = extractTransforms(P)
	# find number of frames 
	N = len(T)
	# declare sum error 
	error = []
	# points in frame 0
	points_0 = x[0]
	# go through each frame of data
	for n in range(N):
		# points in frame n
		points_n = x[n]
		# compute transformed point 
		points_n_rot = rotatePoints(points_n, T[n])
		# compute E_i*p[i]
		error_n = findError(points_0, points_n_rot, P)
		# add current error to sum of errors
		error.extend(error_n)
	# return the total error 	
	return error 

#________________________________________________________
#
# Function: extractTransforms 
# 
# 			given a vector P with transforms for N frames, 
# 			extract the transformation for each frame. Return 
# 			list of transforms 
# 
# \param[in] P: 1xN*6 vector (3 Rodrigues angles, 3 translation variables)
# \return T: 	list of N 4x4 homogeneous transforms 
#---------------------------------------------------------
def extractTransforms(P):
	# assume Rodrigues angle representation 
	varPerFrame = 6
	# find length of P
	numVar = len(P) 
	# compute the number of frames
	if(np.mod(numVar, varPerFrame) != 0):
		print("Error: not even number of variables for each frame!")
		return 	
	# find number of frames 
	N = numVar/ varPerFrame
	# declare list of transforms
	T = []

	# for each frame of parameters
	for frame in range(N):
		# get start and ending indices for P 
		startIndex = frame*varPerFrame
		endIndex = frame*varPerFrame + varPerFrame 
		# print(P[startIndex:endIndex])
		T_frame = formTransform(P[startIndex:endIndex])
		# add current transform to list of transforms
		T.append(T_frame)

	return T

#________________________________________________________
#
# Function: formTransform
# 
# 			given Rodrigues angles and translation, form a homogeneous transform
# 
# \param[in] V: 1x6 array (3 Rodrigues angles, 3 translation)
# \return T: 	4x4 homogeneous transform
#---------------------------------------------------------
def formTransform(V):
	# extract quaternion
	Rod = V[0:3]
	# extract translation 
	translation = V[3:6]
	# make a column vector
	np.reshape(translation, (3,1))
	# form rotation matrix
	R, Jacob = cv2.Rodrigues(Rod)
	# declare homogeneous transform 
	T = np.eye(4)
	# fill with rotation
	T[0:3, 0:3] = R
	# fill with translation
	T[0:3, 3] = translation

	return T
#________________________________________________________

#________________________________________________________
#
# Function: rotatePoints
# 
# 			given a list of 3D coordinates, transform by given 4x4 homogeneous 
#			transform 
# 
# \param[in] points: 	Nx3 array of points 
# \param[in] T:			4x4 homogeneous transform 
# \return points_rot: 	points transformed by T
#---------------------------------------------------------
def rotatePoints(points, T): 
	# point counter
	pt = 0
	worldPoints = np.zeros((3,len(points)))
	# extract rotation 
	R = T[0:3, 0:3]
	# extract translation 
	t = T[0:3, 3:4]
	for point in points:
		# make point a column vector
		point.shape = (3,1)
		# [R|t]*corners_world
		worldPoint = np.dot(R, point) + t
		# make worldPoint a column vector
		worldPoint.shape = (3,1)
		# save worldPoint in matrix worldPoints
		worldPoints[0:3,pt:pt+1] = worldPoint
		# increment pt
		pt = pt + 1
	# take transpose so Nx3 array
	worldPoints = worldPoints.transpose()
	
	return worldPoints

#________________________________________________________
#
# Function: findError
# 
# 			Given 2 pairs of corresponding 3D coordinates, find the rotation and translation error between the point clouds. Weight the rotation and translation errors by the sum of distances between corresponding points in points1 and points2.
#
# \param[in] points1: 	Nx3 array of points 
# \param[in] points2:	Nx3 array of points 
# \return error: 1x6 array of error, sumDistances*[rx ry rz x y z] of rotation between point clouds then translation btwn point clouds
#---------------------------------------------------------
def findError(points1, points2, P):
	mean1 = np.mean(points1, axis=0)
	mean2 = np.mean(points2, axis=0)

	# offset between 2 point clouds 
	t = mean2 - mean1

	# subtact mean off of point clouds
	A = points1 - mean1
	B = points2 - mean2

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
	# eigenvector with the largest eigenvalue
	rotVect = eigVects[:,eigIndex]
	# rotation matrix associated with rotVect
	R = quatToRotM(rotVect)
	# compute the Rodrigues angles of R
	delta_R, Jacob = cv2.Rodrigues(R)
	delta_Rodx = delta_R[0][0]
	delta_Rody = delta_R[1][0]
	delta_Rodz = delta_R[2][0]

	#--------------------------------------
	# 	weight delta error by distance between pt clouds 
	sumError = 0
	
	for (pt1, pt2) in zip(points1, points2):
		distance = dist(pt1, pt2)
		sumError = sumError + distance

	# return the error associated with each parameter
	error = [sumError*delta_Rodx, sumError*delta_Rody, sumError*delta_Rodz, sumError*t[0],  sumError*t[1],  sumError*t[2]]
	
	return error
#_______________________________________________________________________

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
