import cv2
import numpy as np
import os
import pickle
import pdb

from blobDetector import MSERBlobDetector
from markerHistogramExtractor import computeMarkerProb

#_________________________________________________________
#
# Function: blobParamsApp
#
#			let users interactively set blob size parameters for 
#			blob detection in tool tracking 
# \params[in] Folder_Name:		directory with sample images
# \params[in] tool_hist: 		color histogram for tool markers 
# \params[in] N_tool: 			the channel of the probability image with the 
#								tool blob detection
# \return min_area:				int, minimum area of tool marker blobs
# \return max_area: 			int, max area of tool marker blobs
#-----------------------------------------------------------
def createBlobDetectorFile(Folder_Name, tool_hist, N_tool):
	# set parameters for tool markers 
	# number of Gaussian Mixture Model modes for tool prob
	N = 4
	# open the first image in the tool im directory
	for im in os.listdir('experiments/' + Folder_Name + '/Color_Markers/Tool/images/'):
		tool_im = cv2.imread('experiments/' + Folder_Name + '/Color_Markers/Tool/images/' + im, 1)
		break
		
	# apply tool hist to a tool image 
	probImToolAll = computeMarkerProb(tool_im, tool_hist, N, 20)
	# get the right channel of the probability image for tool detection
	probImTool = probImToolAll[N_tool]
	# get min and max areas
	min_area, max_area =  findGoodBlobs(tool_im, probImTool)			
	# save parameters to file 
	blobInfo = [min_area, max_area]
	# save file 
	fileName = 'experiments/' + Folder_Name + '/tool-blob-info.txt'
	with open(fileName, 'wb') as f:
		pickle.dump(blobInfo, f)
	# return min and max areas
	return min_area, max_area

#---------------------------------------------------------------------
# 
# Function: findGoodBlobs
#
#			detect blobs in marker probability image. Let the user change 
#			the desired blob size until all markers are detected. 
# 
# \params[in] image: image from video 
# \params[in] prob_im: tool marker probability image of image
# \return min_area:				int, minimum area of tool marker blobs
# \return max_area: 			int, max area of tool marker blobs
#---------------------------------------------------------------------
def findGoodBlobs(image, prob_im):
	# starting guess for min area
	min_area = 400
	# starting guess for max area
	max_area = 1500
	# area increment change
	delta = 100
	# whether done finding blob params
	done_flag = False 
	# loop until done
	while(not done_flag):
		# detect keypoints with current area parameters
		keypoints = MSERBlobDetector(prob_im, min_area, max_area)
		# draw keypoints on image and prob_im
		blobImage = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		blobProbImage = cv2.drawKeypoints(prob_im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

		outIm = np.concatenate((blobImage, blobProbImage), axis=1)
		# show image and prob_im with blob detections
		cv2.imshow("blobImage", outIm) 
		cv2.waitKey(0)
		# ask for user input
		min_input = input("===== Set Blob min_area ===== \n To decrease min_area size press 1 \n To increase min_area size press 3 \n To keep min_area the same press 0 \n ")


		max_input = input("===== Set Blob max_area ===== \n To decrease max_area size press 1 \n To increase max_area size press 3 \n To keep max_area the same press 0 \n")

		# update min area
		if(min_input > 0):
			print("min_input: " + str(min_input) + " type: " + str(type(min_input)))
			min_area = min_area + delta*(min_input - 2)
			print("new min area: " + str(min_area))
		# update max area
		if(max_input > 0):
			print("max_input: " + str(max_input) + " type: " + str(type(max_input)))
			max_area = max_area + delta*(max_input - 2)
			print("new max area: " + str(max_area))
		# if both are 0, the user is satisfied with the parameters
		if(max_input == 0 and min_input == 0):
			done_flag = True

		# destroy all windows
		cv2.destroyAllWindows()
	# return min and max area
	return min_area, max_area