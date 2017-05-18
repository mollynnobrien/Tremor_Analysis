import cv2
import numpy as np
import os
import pdb

from dist import dist


#______________________________________________________
#
# Function: pruneKeypoints
#
# 			get one keypoint at each location (ignore multiple dections)
# \param[in] keypoints: 	keypoints
# \return good_keypoints:		keypoints, require at distance of at least 2 pixels 
#							between keypoints
#_______________________________________________________
def pruneKeypoints(keypoints):
	# min distance between keypoints in pixels
	good_threshold = 20
	# list of unique keypoints
	good_keypoints = []
	# start by adding first keypoint
	good_keypoints.append(keypoints[0])
	# compare each keypoint
	for keypt in keypoints:
		goodFlag = True
		# to each point already in good_keypoints 
		point = keypt.pt
		for good_kp in good_keypoints:
			good = good_kp.pt
			if(dist(point,good) < good_threshold):
				goodFlag = False
		# if dist > 2 for all points already in good_keypoints
		if(goodFlag):
			# append the keypoint
			good_keypoints.append(keypt)

	return good_keypoints

#_________________________________________________________
#______________________________________________________
# 
# Function: MSERblobDetector
# 
#			find blobs in input image
# \params[in] image: 		probability image 
# \params[in] min_area: 	minimum marker blob area
# \params[in] max_area: 	maximum marker blob area
# \return:	 keypoints:		blob locations and radii in image
#
# Reference: https://www.learnopencv.com/blob-detection-using-opencv-python-c/
#_______________________________________________________
def MSERBlobDetector(image, min_area, max_area):
	# make image 1 channel
	if(len(image.shape) > 2):		
		gray = cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2GRAY)
	else: 
		# gray = np.uint8(image)
		gray = image
	
	
	# smooth image 
	gray = cv2.GaussianBlur(gray, (3,3), 3)
	gray = cv2.GaussianBlur(gray, (5,5), 3)
	gray = cv2.GaussianBlur(gray, (7,7), 3)

	# smooth image more
	gray = cv2.GaussianBlur(gray, (3,3), 2)
	retval, gray = cv2.threshold(gray, 150, 255, 0)

	# create MSER detector
	mser = cv2.MSER_create()
	# set min blob area 
	mser.setMinArea(min_area)
	# set max blob area 
	mser.setMaxArea(max_area)
	print("image.shape: " + str(image.shape))
	
	# detect regions in image
	regions = mser.detect(gray)
	
	if len(regions) > 0:
		# only get one keypoint at each location
		keypoints = pruneKeypoints(regions)
	else:
		keypoints = regions

	points = []
	for kp in keypoints:
		points.append(kp.pt)


	return keypoints

#______________________________________________________
# 
# Function: MSERblobDetector
# 
#			find blobs in input image
#
# \params[in] image: 	probability image 
# \params[in] key: 		string indicating "tool" or "marker"
# \return 	keypoints: 	blob keypoints
# \return	points:		center of blob locations in image
# \return 	blobImage: 	image with detected blobs drawn on image
#
# Reference: https://www.learnopencv.com/blob-detection-using-opencv-python-c/
# #_______________________________________________________
def MSERBlobDetectorDebugging(image, key):
	# convert image to grayscale if color image
	if(len(image.shape) > 2):		
		image = cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2GRAY)


	kernel3 = np.ones((3,3),np.uint8)
	kernel5 = np.ones((5,5),np.uint8)
	kernel9 = np.ones((9,9),np.uint8)
	kernel21 = np.ones((21,21),np.uint8)
	gray = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel5)
	gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel5)
	gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel9)
	gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel9)
	gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel9)
	gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel9)
	# smooth image 
	gray = cv2.GaussianBlur(gray, (3,3), 3)
	gray = cv2.GaussianBlur(gray, (5,5), 3)
	gray = cv2.GaussianBlur(gray, (7,7), 3)

	# do closing to close holes in connected component
	gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel5)
	gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel5)
	gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel9)
	gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel9)

	# create MSER detector
	mser = cv2.MSER_create()
	# pdb.set_trace()
	if key == "marker":
		# # # set max blob area 
		mser.setMaxArea(2200)

	elif key == "tool":
		gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel9)
		gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel21)
		gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel21)
		gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel9)

		# smooth image more
		gray = cv2.GaussianBlur(gray, (3,3), 2)
		retval, gray = cv2.threshold(gray, 150, 255, 0)
		mser.setMinArea(200)
		mser.setMaxArea(800)

	# elif key == "marker_template":


	print("image.shape: " + str(image.shape))

	# detect regions in image
	keypoints = mser.detect(gray)

	# copy the image to draw on
	blobImage = image.copy()
	if(len(keypoints) > 0):
		keypoints = pruneKeypoints(keypoints)

	blobImage = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	points = []
	for kp in keypoints:
		points.append(kp.pt)

	    	# return blobImage, keypoints, points
    	return keypoints, points, blobImage
#______________________________________________________
