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
sys.path.append('home/molly/opencv2')

#___________________________________________________________________
#
# Function: readInCalIms
# 	    	read in images for stereo calibration from left_im_folder and 
#			right_im_folder
#
# Author:   Molly O'Brien
# Input: 	left_dir: folder containing left images 
#			right_dir: folder containing right images
# 			output: file where results & debugging info are saved
# Return:	left_ims: list with left calibration images 
#			right_ims: list with right calibration images
# 			image_list: list of calibration image names
#------------------------------------------------------------------- 
def readInCalIms(left_dir, right_dir, output):
	print("In read in cal ims")
	# list of left images 
	left_ims = []
	# list of right images
	right_ims = []

	# files are named by frame number in left and right directory
	# both directories have the same images with same file names
	im_num = 0
	# image_dict[image name] = number image is in list
	image_list = []
	for frame in os.listdir(left_dir):
		# write name to output file
		output.write("Image " + str(im_num) + " is " + frame + '\n')
		num, jpeg = frame.split('.')
		# get the image frame number
		num_int = int(num)
		output.write("Frame number" + str(num_int) + '\n')

		# save the image number with the order it was found 
		image_list.append(num_int)

		# read file from left im 
		LIM = cv2.imread(left_dir + '/' + frame)
		# convert image to grayscale
		GLIM = cv2.cvtColor( LIM, cv2.COLOR_BGR2GRAY);
		# add to list of left images
		left_ims.append(LIM)
		# read corresponding right image
		RIM = cv2.imread(right_dir + '/' + frame)
		# convert to grayscale
		GRIM = cv2.cvtColor( RIM, cv2.COLOR_BGR2GRAY);
		# add to list of right images
		right_ims.append(RIM)
		# increment image counter
		im_num = im_num + 1
				
	return left_ims, right_ims, image_list

#--------------------------------------------

