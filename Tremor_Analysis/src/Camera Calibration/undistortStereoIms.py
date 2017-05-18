import cv2
import numpy as np
import sys
import math
import os
from numpy.linalg import inv
import random
import re
import pickle

from loadParams import loadCamCal, loadSterCal
#_______________________________________________________________________
#
# Function: undistortStereoIms():
# 		given camera calibration parameters, remove lense distortion
#
# Author: 	Molly O'Brien
# Input: 	sterCalFile: stereo calibration filename 
#			imageL: left image with lense distortion
#			imageR: right image with lense distortion 
# Return: 	undistortL: left image with lense distortion removed
# 			undistortR: right image with lense distortion removed 
#-----------------------------------------------------------------------
def undistortStereoIms(sterCalFile, imageL, imageR):
	# read in stereo calibration parameters
	retval, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F = loadSterCal(sterCalFile)

	# declare variable for undistorted image
	undistortL = None
	# undistort left image
	undistortL = cv2.undistort(imageL, cameraMatrixL, distCoeffsL)

	# declare variable for undistorted image
	undistortR = None
	# undistort right image
	undistortR = cv2.undistort(imageR, cameraMatrixR, distCoeffsR)

	# display images
	showFlag = False 
	if(showFlag):
		cv2.imshow("Left undistort", undistortL)
		cv2.waitKey(0)

		cv2.imshow("Right undistort", undistortR)
		cv2.waitKey(0)

	return undistortL, undistortR
#________________________________________________________________________
