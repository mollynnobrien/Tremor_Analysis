import cv2
import numpy as np 
import os
import pickle
import sys
import pdb


from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from blobDetector import MSERBlobDetector

#_____________________________________________________________________
#
# Function: markerHistogramExtractor
#
#			given images, binary masks for the markers compute a color histogram
#			for the color marker
# \param[in] image dir: 	directory where reference marker images are 
# \param[in] mask dir: 		directory where binary marker masks are 
# \return H_hist: 			histogram in Hue space
# \return S_hist: 			histogram in Saturation space
# 
# References: http://matplotlib.org/examples/pylab_examples/hist2d_log_demo.html
#_______________________________________________________________________
def markerHistogramExtractor(image_dir, mask_dir):#, save_dir, output): 
	print("image directory: " + str(image_dir))
	# list of Hue images
	H_list = []
	# list of Saturation images
	S_list = []
	# image counter 
	numIms = 0
	# for each image in the image directory
	for imName in os.listdir(image_dir):
		print(imName)
		# open the image 
		image = cv2.imread(image_dir + imName)		
		# get the corresponding binary mask
		binaryImName = mask_dir + imName
		# open the binary mask
		binaryMask = cv2.imread(binaryImName)
		# apply the mask
		maskedIm = image.copy()
		maskedIm[:,:,0] = cv2.bitwise_and(image[:,:,0],binaryMask[:,:,0])
		maskedIm[:,:,1] = cv2.bitwise_and(image[:,:,1],binaryMask[:,:,0])
		maskedIm[:,:,2] = cv2.bitwise_and(image[:,:,2],binaryMask[:,:,0])
		# save the masked image
		# cv2.imwrite(save_dir+imName, maskedIm)
		#-------------------------------------------------------
		# find histogram
		# convert masked image into HSV 
		hsv = cv2.cvtColor(maskedIm, cv2.COLOR_BGR2HSV)
		# extract hue image, change range from [0, 180]-> [0, 255]
		H_im = hsv[:,:,0]
		# append to hue list
		H_list.append(H_im)
		# extract saturation image
		S_im = hsv[:,:,1]
		# append to saturation list
		S_list.append(S_im)
		# increment numIms
		numIms = numIms + 1

	# create rows x cols x numIms Hue and Sat images 
	rows,cols,channels = image.shape
	# declare arrays
	HueArray 	= np.zeros((rows,cols,numIms))
	SatArray 	= np.zeros((rows,cols,numIms))
	
	# iterate through each image 
	for i in range(numIms):
		HueArray[:,:,i] = H_list[i]
		SatArray[:,:,i] = S_list[i]

	# take 2D histogram of H & S channel
	HS_Hist, xedges, yedges = np.histogram2d(HueArray.ravel(), SatArray.ravel(),[180, 256],[[0,179],[0,255]])
	plt.hist2d(HueArray.ravel(), SatArray.ravel(),[180, 256],[[0,179],[0,255]], norm=LogNorm())
	plt.colorbar()
	# plt.show()

	# create vector of (H,S) pairs in segmented marker ims, ignore pixels where 
	# H&S = - 0, pixels set to 0 by binary mask 
	hue1D = HueArray.ravel()
	sat1D = SatArray.ravel()

	samples1D 		= np.zeros((len(hue1D), 2))
	sampCounter 	= 0
	for h, s in zip(hue1D, sat1D): 
		# if h or s != 0
		if(h != 0 or s != 0):
			# save values to array
			samples1D[sampCounter, :] = [h, s]
			sampCounter = sampCounter + 1

	print(str(sampCounter) + " non-zero (H,S) pairs")
	# get only non-zero rows
	samples1D = samples1D[0:sampCounter, :]
	print("samples1D: " + str(samples1D))

	# output.write("samples1D: \n")
	# output.write(str(samples1D) + '\n') 

	return samples1D

#__________________________________________________________________________

#__________________________________________________________________________
#
# Function: computeGaussMixModel(N, Hist)
#
#			given a histogram, compute a Gaussian Mixture Model with N clusters
# \param[in] N: 	int, number of clusters
# \param[in] Hist: 	histogram computed with np.histogram2d
# \return: 	em: 	Gaussian mixture model 
#
# Reference: http://answers.opencv.org/question/66881/opencv3-expectation-maximization-python-getcovs/
#--------------------------------------------------------------------------
def computeGaussMixModel(N, samples): #, output):
	# OpenCV's Expected Maximization Algorithm
	em = cv2.ml.EM_create()
	# set number of clusters
	em.setClustersNumber(N)

	# train EM 
	retval, logLikelihoods, labels, probs = em.trainEM(samples)
	print("retval: " + str(retval))
	print("logLikelihoods: " + str(logLikelihoods))
	print("labels: " + str(labels))
	print("probs: " + str(probs))

	return em

#__________________________________________________________________________

#__________________________________________________________________________
#
# Function: computeMarkerProb(image_dir, gaussMixModel)
# 			
#			given images, hue and saturation histograms of markers, compute the 
#			the probability of each pixel in images of being a marker 
# \param[in] image_dir: 		directory where images live
# \param[in] GMM: 				GaussMixModel with prob pixel val in H&S was a 
#								marker 
# \param[in] N:					number of modes in GMM
# \param[in] val_thresh:		min value required to be considered to be a marker, 
#								needed for blue markers because low val, blue areas
# 								are often shadows
# \return:	probIm:				binary image, white pixel = high marker prob
#___________________________________________________________________________
def computeMarkerProb(image, HS_em, N, val_thresh):
	# convert image to hsv
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# extract hue image
	hueIm = hsv[:,:,0]
	# extract sat image
	satIm = hsv[:,:,1]
	# extract value image
	valIm = hsv[:,:,2]
	# get mask for pixels not in shadow
	retval, valueMask_low = cv2.threshold(valIm, val_thresh, 255, 0)
	retval, valueMask_high = cv2.threshold(valIm, 200, 255, 1)
	valueMask = cv2.bitwise_and(valueMask_low, valueMask_high)
	# ravel images into 1D vectors
	hueIm1D = hueIm.ravel()
	satIm1D = satIm.ravel()

	# create Nsamplesx2 array of H,S values 
	HSpairs = np.zeros((len(hueIm1D), 2))
	HSpairs[:,0] = hueIm1D
	HSpairs[:,1] = satIm1D

	# make empty array for probability image
	probIm1D_cl1 = np.zeros(hueIm1D.shape, np.float32)
	probIm1D_cl2 = np.zeros(hueIm1D.shape, np.float32)
	probIm1D_cl3 = np.zeros(hueIm1D.shape, np.float32)
	# pixel counter 
	pixCounter = 0
	# compute the marker probability at each pixel
	retval, results = HS_em.predict(HSpairs)

	probIm = []
	# get size of image 
	rows, cols, channels = image.shape	
	# flag to save images
	writeFlag = True
	# extract images from the different GMM modes
	for mode in range(N):
		# pdb.set_trace()
		probVect = 255 * results[:,mode]
		# reshape vector into an image
		probIm1Channel = probVect.reshape(rows, cols)
		# make image
		probImage = image.copy()
		
		# get black and white image with RBG channels, apply value mask		
		probImage[:,:,0] = cv2.bitwise_and(np.float64(valueMask), probIm1Channel)
		probImage[:,:,1] = cv2.bitwise_and(np.float64(valueMask), probIm1Channel)
		probImage[:,:,2] = cv2.bitwise_and(np.float64(valueMask), probIm1Channel)

		# save probability image to list
		probIm.append(probImage)
		# save image if flag
		if(writeFlag):
			cv2.imwrite("probIm_" + str(mode) + ".jpg", probImage)
	
	return probIm
#__________________________________________________________________________

#__________________________________________________________________________
#
# Function: markerMatch(image, markerIm, markerPts)
#
# 			match a marker template image to a marker probability image. 
#			Return location of all marker keypoints. Want to check possible 
#			scale changes and rotations 
# 
# \params[in] image: 	binary probability image, white pixels = high marker prob
# \params[in] templateIm: 	binary image with ground truth indv marker positions.  The
#						marker is rigid so the relative pos of indv markers is
#						constant
# \params[in] templatePts:	pixel locations of centers of indv markers in markerIm 
# \params[in] prevEst: boolean flag, are there previous match parameters?
# \params[in] prevParams: translation and rotation of marker match in previous frame
# \return 	imagePts: 	pixel locations of centers of indv markers in image
# \return 	params: translation and rotation of marker match
#----------------------------------------------------------------------------
def markerMatch(image, templateIm, templatePts, prevEst, prevParams):
	# get size of template image
	temp_rows, temp_cols = templateIm.shape
	image = np.float32(image)
	templateIm = np.float32(templateIm)
	# scale templateIm to be between -100 and 100
	templateIm = templateIm*200.0# - 100
	
	# since template is taken from image, scale is 1
	scale = 1.0
	# define center of rotation, top left corner #use center of template 
	center = (0,0)#(np.int(np.floor(temp_rows/2)), np.int(np.floor(temp_cols/2)))
	# define scales 
	if(prevEst):
		prevScale = prevParams[0]
		prevAngle = prevParams[1]
		# use smaller search range around previous estimate
		rot_angles = range(-2 + 10*prevAngle, 2 + 10*prevAngle)
		rot_angles = [r/10 for r in rot_angles]
	else:
		# larger search range for first guess
		# define rotations 
		rot_angles = range(359)
	# declare max match value
	maxMatchVal = 0
	# declare best rotation 
	bestR = []
	# declare best translation
	bestT = []
	# declare best angle
	best_angle = None
	# declare best scale 
	best_scale = None

	# go through all pos template rotations 
	# for scale in scales: 
		# print("in scale " + str(scale))
	for angle in rot_angles: 
		# rotate template by angle
		rotationMatrix = cv2.getRotationMatrix2D(center, angle, scale)
		# pdb.set_trace()
		new_template = cv2.warpAffine(templateIm, rotationMatrix, (templateIm.shape[1], templateIm.shape[0]))

		# template match. Method 4 -> NCC
		result = cv2.matchTemplate(image[:,:,0], new_template, 4)
		# find the location of the max value in result
		minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)		
		# scale maxVal by maxVal
		maxVal = maxVal/ scale
		
		# if this match is better than the prev best
		if(maxVal > maxMatchVal):
			print("maxVal: " + str(maxVal) + "at " + str(maxLoc))
			# update best transform 
			bestR = rotationMatrix[0:2, 0:2]
			bestT = np.array([[maxLoc[0]], [maxLoc[1]]]) + rotationMatrix[0:2, 2:3]
			bestT.shape = (2,1)
			best_scale = scale 
			best_angle = angle
			maxMatchVal = maxVal

	print("best scale: " + str(best_scale))
	print("best angle: " + str(best_angle))

	# rigid transform = best transform 
	imagePts = rotatePoints2D(templatePts, bestR, bestT)

	for pt in imagePts: 
		cv2.circle(image, (int(pt[0]), int(pt[1])), 3, (255,0,0), -1)
	
	return imagePts, [best_scale, best_angle]

#___________________________________________________________________

#___________________________________________________________________
# 
# Function: rotatePoints2D(points, E)
# 	
#			transform 2D points
# 
# \params[in] points: array of 2D points
# \params[in] R: 2D rotation
# \params[in] t: 2D translation 
# \return: worldPoints: transformed points
#--------------------------------------------------------------------
def rotatePoints2D(points, R, t):
	#print("in rotate points")
	pt = 0
	worldPoints = np.zeros((2,len(points)))
	for point in points:
		pointArray = np.array([point[0], point[1]])
		# make point a column vector
		pointArray.shape = (2,1)
		# [R_world_cam|t]*corners_world
		worldPoint = np.dot(R, pointArray) + t
		# make cameraPoint a column vector
		worldPoint.shape = (2,1)
		# save cameraPoint in matrix cameraPoints
		worldPoints[0:2,pt:pt+1] = worldPoint
		# increment pt
		pt = pt + 1

	worldPoints = worldPoints.transpose()
	
	return worldPoints
#___________________________________________________________________

