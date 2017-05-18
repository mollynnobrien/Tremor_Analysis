import cv2
import numpy as np 
import os
import sys
import random
import pdb

sys.path.insert(0, 'src/Camera Calibration/')

from blobDetector import MSERBlobDetector
from dist import dist
from markerHistogramExtractor import computeMarkerProb

#____________________________________________________________________
# 
# Function: blobMatch
# 	
#			given 2 images, find blobs, match between the frames 
# 
# \params[in] frame1: 	image 1
# \params[in] frame2: 	image 2
# \params[in] HistInfo:	list of color marker histogram info. For each type of color marker the list has [color marker histogram, number of Gaussian Mixture Model modes, probability image channel for detections, histogram string key, (if tool min blob detection area, max blob detection area)].
# \params[in] frameNum: frame number
# \return matches1:		array of feature point locations in image 1
# \return matches2: 	corresponding array of feature point locations in image 2
# \return kp1:		array of keypoints in image 1
# \return kp2: 		corresponding array of keypoints in image 2
#-----------------------------------------------------------------
def blobMatch(frame1, frame2, HistInfo, frameNum):
	print("in blobMatchDebug")
	# extract histogram
	HS_em = HistInfo[0]
	# extract GMM modes
	N = HistInfo[1]
	# which channel gives good marker prediction?
	N_select = HistInfo[2]
	# extract good GMM mode
	key = HistInfo[3]
	min_area = HistInfo[4]
	max_area = HistInfo[5]

	if key == "background": 
		val_Thresh = 60
	elif key == "tool":
		val_Thresh = 20

	else:
		print("Error: Key " + key + " not background, or tool.") 
		

	# compute marker prob image for frame 1
	probIm1All = computeMarkerProb(frame1, HS_em, N, val_Thresh)
	# compute marker prob image for frame 2
	probIm2All = computeMarkerProb(frame2, HS_em, N, val_Thresh)

	probIm1 = np.zeros(probIm1All[0].shape)
	probIm2 = np.zeros(probIm1All[0].shape)
	
	if(isinstance(N_select, int)):
		# if it is just an int do this:
		probIm1 = probIm1All[N_select]
		probIm2 = probIm2All[N_select]
	else: 		
		# if N_select is a list do this:
		for channel in N_select:
			probIm1 = probIm1 + probIm1All[channel]
			probIm2 = probIm2 + probIm2All[channel]
			
	# detect blobs in frame1
	keypoints1 = MSERBlobDetector(probIm1, min_area, max_area)
	# detect blobs in frame2
	keypoints2 = MSERBlobDetector(probIm2, min_area, max_area)
	# match keypoints using RANSAC
	matches1, matches2, matchKp1, matchKp2 = findMatches(keypoints1, keypoints2, 60)

	return matches1, matches2, matchKp1, matchKp2


#_________________________________________________________________


#_________________________________________________________________
#
# Function: findMatches
# 	    Match marker feature points between left and right images
#
# Author: Molly O'Brien 
# Inputs: points1: feature points in frame 1
#         points2: feature points in frame 2
#		  d: max distance allowed between matching feature point locations in 2 images 
# Return: matches1: array with points in points1 st F*points1 - points2 < d
#         matches2: array with points in points2 st F*points2 - points2 < d
# \return matchKp1:	array of keypoints in image 1
# \return matchKp2: corresponding array of keypoints in image 2
#-----------------------------------------------------------------
def findMatches(points1, points2, d):
	# lists that will hold the inliers
	matches1 = []
	matches2 = []
	matchKp1 = []
	matchKp2 = []

	# for pt in points1
	for keypt1 in points1:
		# make a copy of points2 (bc we will remove the points that are matched from this list)
		keypoints2 = points2
		# get keypoint location
		pt1 = keypt1.pt
		# apply transform
		# look through points2
		for keypt2 in keypoints2:
			# get point location 
			pt2 = keypt2.pt
			print("distance: " + str(dist(pt1, pt2)))

			if(dist(pt1, pt2) < d):
				# add pt1 to matches 
				matches1.append(pt1)
				matches2.append(pt2)
				matchKp1.append(keypt1)
				matchKp2.append(keypt2)
				# remove keypt2 from keypoints2 
				keypoints2.remove(keypt2)

	return matches1, matches2, matchKp1, matchKp2		
