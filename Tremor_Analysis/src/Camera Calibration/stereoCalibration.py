import cv2
import numpy as np
import sys
import math
import os
from numpy.linalg import inv
import random
import re
import pickle

# import my functions
from loadCheckerBoard import loadCheckerBoard
from dist import dist
from readInCalIms import readInCalIms

#___________________________________________________________________
#
# Function: stereoCalibration
# 	    	Given left and right images from a stereo camera find the 
# 	    	stereo calibration and stereo rectification matrices 
#
# Author:   	Molly O'Brien
# Input: 	left_dir: folder with left calibration images 
#	 		right_dir: folder with right calibration images
#			camCalFileName: file name to save calibration results
#			stereoCalFileName: file name to save stereo calibration results
#			output: file where results & debugging info are saved 
# Return:	ObjectPoints: the 3D checkerboard corner positions used in calibration 
# 			leftCalIms: the left images were the L & R ims had checkerboards detected ( and were used in calibration)
#			rightCalIms: the right images that were used in calibration 
# References: http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.stereoCalibrate 
# http://stackoverflow.com/questions/29628445/meaning-of-the-retval-return-value-in-cv2-calibratecamera)
#------------------------------------------------------------------- 

def stereoCalibration(left_dir, right_dir, camCalFileName, stereoCalFileName, output):
	# read in calibration images 
	# leftIms: list of left calibration images. leftIms[0]: image 1
	# rightIms: list of right calibration images
	leftIms, rightIms, image_list = readInCalIms(left_dir, right_dir, output)
	# calibration image list: store image frame number of the images where the corners were found
	image_cal_list = []
	# find the size of the images
	numIms = len(leftIms)
	row, col, channels = leftIms[0].shape

	#-----------------------------------------------------
	# empty array that will hold pixel position of detected corners in left cal images
	leftPoints = []
	# empty array that will hold pixel position of detected corners in right cal images
	rightPoints = []

	# lists to hold the images used in calibration 
	leftCalIms = []
	rightCalIms = []

	# load world coordinate points of checkerboard corners
	objectPoints, col_corn, row_corn = loadCheckerBoard(2, numIms)

	# set the number of internal corners
	patternsize = (row_corn, col_corn)

	numFoundIms = 0
	# find corners in the images
	for image in range(numIms):
		# detect corners
		foundL, cornersL, foundR, cornersR = findCorners(leftIms[image], rightIms[image], patternsize)

		# display corners
		drawChessboardCorners = False
		if(drawChessboardCorners): 
			cornersLeftIm, cornersRightIm = drawCorners(leftIms[image], rightIms[image], cornersL, cornersR, patternsize, foundL, foundR, True)  
			destL = str(numFoundIms) + "L.jpg"
			destR = str(numFoundIms) + "R.jpg"

			cv2.imwrite(destL, cornersLeftIm)
			cv2.imwrite(destR, cornersRightIm)


		if(foundL and foundR):
			# save frame number of image that was found (will be used to know what image the R, t found in calibrateCamera correspond to)
			image_cal_list.append(image_list[image])
			# write the image was found to output file
			output.write("Image " + str(image) + "found!" + '\n')
			# save corner positions in left image
			leftPoints.append(cornersL)

			# save right corners in vector		
			rightPoints.append(cornersR)

			leftCalIms.append(leftIms[image])
			rightCalIms.append(rightIms[image])

			numFoundIms = numFoundIms + 1

	if(numIms != numFoundIms):
		# load world coordinate points of checkerboard corners
		objectPoints, col_corn, row_corn = loadCheckerBoard(2, numFoundIms)

	# calibrate left camera 
	foundL, cameraMatrixL, distCoeffsL, rvecsL, tvecsL = cv2.calibrateCamera(objectPoints, leftPoints, (col, row), None, None)
	# foundL is the final reprojection error between the objectPoints and the image corners
	#print("foundL " + str(foundL))	

	# calibrate right camera 
	foundR, cameraMatrixR, distCoeffsR, rvecsR, tvecsR = cv2.calibrateCamera(objectPoints, rightPoints, (col, row), None, None)

	#--------------------------------------------------------------
    # Save calibration results
    # add calibration results to one list
   	calResults = []
   	calResults.append(foundL)
   	calResults.append(rvecsL)
   	calResults.append(tvecsL)
   	calResults.append(foundR)
   	calResults.append(rvecsR)
   	calResults.append(tvecsR)
   	calResults.append(image_cal_list)

    # save the list 
   	with open(camCalFileName, 'wb') as f:
            pickle.dump(calResults, f)

	#-------------------------------------------------------------
	# for stereoCalibrate

	# declare variable for stereoCalibrate
	R = None
	T = None
	E = None
	F = None

	# stereo calibration, find calibration for left and right cameras

	#help(cv2.stereoCalibrate)
	retval, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F = cv2.stereoCalibrate(objectPoints, leftPoints, rightPoints, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR,(col, row), R, T, E, F)

	#cv2.stereoCalibrate(objectPoints1Frame, cornersL, cornersR, (row,col), cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F)

	# objectPoints: points on the calibration object in world coordinates
	#               for me this is checkerboard corner positions 
	#               (x,y,0)
	# imagePoints: detected checkerboard corner positions in the image
	# retval: RMS reprojection error (for a good calibration it should be ~ 0.1-1 pixels) 
		# cameraMatrix: A = [f_x 0 c_x; o f_y c_y; 0 0 1]
	# f_x, f_y: x/y focal length
	# (c_x, c_y): principle pt (center of image)
	# distCoeffs: distortion coefficients (k1, k2, p1, p2, k3, k4, k5, k6) 
	#       k's: radial distortion, p's: tangential distortion
	# R: rotation matrix btwn 1st and 2nd image (L and R image)
	# T: translation btwn L & R camera coordinate systems
	# E: essential matrix
	# F: fundamental matrix

	#--------------------------------------------------------------
	# Save calibration results	
	# add calibration results to one list
	sterResults = [] 
	sterResults.append(retval)
	sterResults.append(cameraMatrixL)
	sterResults.append(distCoeffsL)
	sterResults.append(cameraMatrixR)
	sterResults.append(distCoeffsR)
	sterResults.append(R)
	sterResults.append(T)
	sterResults.append(E)
	sterResults.append(F)


	print("cameraMatrixR" + str(cameraMatrixR))
	print("distCoeffsR" + str(distCoeffsR))

	# save the list 
	with open(stereoCalFileName, 'wb') as f:
		pickle.dump(sterResults, f)

	return objectPoints, leftCalIms, rightCalIms
#_______________________________________________________________________

#___________________________________________________________________
#
#_______________________________________________________________________
# 
# Function: findCorners 
#			find corners in stereo ims of a calibration checkerboard
# Author: Molly O'Brien 
# Input: 	leftIm: left image
#		  	rightIm: right image
#			patternsize: (corners along cols, corners along rows)
# Return: 	foundL: True/False if all corners were found
#			cornersL: corners on left image
#			foundR: True/False if all corners were found
#			cornersR: corners on right image
#			* will return None,None if can't find full checkerboard in 
#			both images
#------------------------------------------------------------------------
def findCorners(leftIm, rightIm, patternsize):
	# find checker board corners in the left image
	foundL, cornersL = cv2.findChessboardCorners(leftIm, patternsize)

	# find checker board corners in the right image
	foundR, cornersR = cv2.findChessboardCorners(rightIm, patternsize)

	# print whether checkerboard detection was successful
	if(foundL):
		print("Left checkerboard found")
    
   	else:
   		print("Left checkerboard not found")      

	if(foundR):
		print("Right checkerboard found")
   	else:
   		print("Right checkerboard not found")

	return foundL, cornersL, foundR, cornersR
#_______________________________________________________________________