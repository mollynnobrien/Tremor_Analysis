import numpy as np
import sys
import math
import os
import cv2
import pickle
import pdb

from matplotlib import pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation

sys.path.insert(0, 'src/Color Marker Detection/')
sys.path.insert(0, 'src/Bundle Adjustment/')
sys.path.insert(0, 'src/Camera Calibration/')
sys.path.insert(0, 'src/Frequency Analysis/')

from undistortStereoIms import undistortStereoIms
from triangulate import triangulate
from ICP import findTransform, stabilizePoints
from computeCameraCalibration import computeCameraCalibration
from blobMatch import blobMatch
# from blobDetector import MSERBlobDetector
from markerHistogramExtractor import markerHistogramExtractor, computeGaussMixModel, computeMarkerProb, markerMatch
from bundleAdjustment import bundleAdjustment

#___________________________________________________________________
#
# Function: computeCameraMovement
# 	    	* read in calibration files 
#			* take in left and right videos
#			* detect color markers, match in left and right frames
#			* triangulate a 3D feature point cloud for each frame
#			* estimate the 3D transform between each frame  
#
# Author:   	Molly O'Brien
# Input: 	capLeft: left video
#			capRight: right video 
#			camCalFile: camera calibration file
#			sterCalFile: stereo calibration file
# 			outputFile: text file to write program info to
#			trialName: experiment name
#			outL: left output video
#			outR: right output video
#			template_L: background marker template for left video
#			template_points_L: center of background markers in left template image 
#			template_R: background marker template for right video
#			template_points_R: center of background markers in right template image 
# Return:	marker_points3D: list of 3D background points in each video frame
#			tool_points3D: list of 3D tool points in each video frame 
#			frameTransform: 4x4 homogeneous transform from each frame to the first frame
#------------------------------------------------------------------- 
def computeCameraMovement(capLeft, capRight, HistInfo, camCalFile, sterCalFile, outputFile, trialName, outL, outR, template_L, template_points_L, template_R, template_points_R):
	# initialize frameCount
	frameCount = 0
	# declare frame transform
	frameTransform = []
	# list for 3D points marker points in each frame
	marker_points3D = []
	# list for 3D points tool points in each frame
	tool_points3D = []
	# list of 2D detections 
	marker_matches_list = []
	# bool for markerMatch in triangulate_marker_points 
	prevFrame = False
	# start with no prev markers params
	prevMarkerParams = []
	# iterate through frames findTransform(points_n, points_0)in video
	while(capLeft.isOpened() and capRight.isOpened()):
		print("frameCount: " + str(frameCount))
		
		outputFile.write("\n \n _______________________________________ \n \n")
		outputFile.write("frameCount: " + str(frameCount))

		# read left and right images 
		retL, frameDistL = capLeft.read()
		retR, frameDistR = capRight.read()
		if(retL and retR):
			# undistort images 
			frameL, frameR = undistortStereoIms(sterCalFile, frameDistL, frameDistR)

			# triangulate marker and tool points 
			marker_points, tool_points, prevMarkerParams, marker_matches = triangulate_marker_points(frameL, frameR, HistInfo, camCalFile, sterCalFile, outL, outR, template_L, template_points_L, template_R, template_points_R, frameCount, prevMarkerParams, trialName)
			
			if(frameCount > 0):
				# find the rotation between current frame and frame 0
				T = findTransform(marker_points, marker_points_0)
				
			else:
				# the transform between the first frame and itself is identity
				T = np.eye((4))
				# the first frame of marker points 
				marker_points_0 = marker_points 
				# now previous frame
				prevFrame = True
							
			#----------------------------------------------------------		
			marker_points3D.append(marker_points)
			tool_points3D.append(tool_points)

			frameTransform.append(T)
			marker_matches_list.append(marker_matches)
			# increment frameCount 
			frameCount = frameCount + 1

			# every 100 frames save points in case there is an error 
			if(np.mod(frameCount, 100) == 0):
				# save results in a list
				computeCamMotResults = [marker_points3D, tool_points3D, frameTransform, marker_matches_list]
				# save list
				with open(trialName+"Tand3Dpoints.txt", 'wb') as f:
					pickle.dump(computeCamMotResults, f)
		else: 
			break

	# save the final triangulated points
	computeCamMotResults = [marker_points3D, tool_points3D, frameTransform, marker_matches_list]
	# save list
	with open(trialName+"Tand3Dpoints.txt", 'wb') as f:
		pickle.dump(computeCamMotResults, f)
		
	return marker_points3D, tool_points3D, frameTransform

#___________________________________________________________________

#___________________________________________________________________
#
# Function: triangulate_marker_points
# 	    	* read in left and right images
#			* compute marker probability images for each diff kind of marker
#			* match background marker to template
# 			* detect centers of other markers 
#			* return 3D locations of markers
#
# Author:   	Molly O'Brien
# Input: 	frameL: left image
#			frameR: right image 
# 			HistInfo: color histogram info for each color marker
#			camCalFile: camera calibration file
#			sterCalFile: stereo calibration file
# 			outputFile: text file to write program info to
#			trialName: experiment name
#			outL: left output video
#			outR: right output video
#			template_L: background marker template for left video
#			template_points_L: center of background markers in left template image 
#			template_R: background marker template for right video
#			template_points_R: center of background markers in right template image 
#			frameCount: 	current frame number
#			prevMarkerParams: the rotation and translation of the background markers in the previous frame
# Return:	marker_points: 	3D background marker positions
#			tool_points: 	3D tool marker positions
#			bestMarkerMatchParams: the best rotation and translation of the background markers in the left and right frames
#			marker_matches: background marker points in the left and right frames
#------------------------------------------------------------------- 
def triangulate_marker_points(frameL, frameR, HistInfo, camCalFile, sterCalFile, outL, outR, template_L, template_points_L, template_R, template_points_R, frameCount, prevMarkerParams, trialName):
	# will draw detections on copy of frameL and frameR
	outImL = frameL.copy()
	outImR = frameR.copy()
	prevFrame = frameCount > 0
	# extract features, match from L to R
	for histList in HistInfo:
		# get the marker detection info
		hist_em = histList[0]
		# number of modes in GMM
		N = histList[1]
		# select channel
		N_select =  histList[2]
		# key describing markers, "marker" or "tool"
		key = histList[3]

		if(prevFrame):
			# extract prev params
			prevParamsL = prevMarkerParams[0]
			prevParamsR = prevMarkerParams[1]
		else:
			prevParamsL = []
			prevParamsR = []
		
	 	# key specific stuff
	 	if key == "background":
	 		# min gray level value
	 		val_Thresh = 60
	 		probImLAll = computeMarkerProb(frameL, hist_em, N, val_Thresh)
	 		probImRAll = computeMarkerProb(frameR, hist_em, N, val_Thresh)
	 		# channel we want for marker detection 
	 		probImL = probImLAll[N_select]
	 		probImR = probImRAll[N_select]
	 		# if first frame save the left and right background probability images
	 		if(frameCount == 0):
	 			cv2.imwrite("left_template.jpg", probImL)
	 			cv2.imwrite("right_template.jpg", probImR)
	 		# find template in image 
	 		# match marker template to probability image, always listed in same order so don't need to match	 		
			matchesL, bestParamsL = markerMatch(probImL, template_L, template_points_L, prevFrame, prevParamsL)
			matchesR, bestParamsR = markerMatch(probImR, template_R, template_points_R, prevFrame, prevParamsR)
	 		# drawing color
	 		color = (255, 255, 0)
	 		# triangulate 3D background fiducial points 
			marker_points = triangulate(camCalFile, sterCalFile, matchesL, matchesR)
			# save image detection points
			marker_matches = [matchesL, matchesR]
			
	 	elif key == "tool":
	 		# find marker matches between the two frames
			matchesL, matchesR, Kp_L, Kp_R = blobMatch(frameL, frameR, histList, frameCount)		
			# drawing color
	 		color = (0, 255, 0)
				# triangulate 3D tool fiducial points
			tool_points = triangulate(camCalFile, sterCalFile, matchesL, matchesR)

		# draw detections on image
		for ptL, ptR in zip(matchesL, matchesR):
			outImL = cv2.circle(outImL, (int(ptL[0]), int(ptL[1])), 5, color, -1)
			outImR = cv2.circle(outImR, (int(ptR[0]), int(ptR[1])), 5, color, -1)
		# outIm = cv2.drawKeypoints(outIm, Kp_L, np.array([]), color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	
	outL.write(outImL)
	outR.write(outImR)
	# return triangulated marker points
	return marker_points.transpose(), tool_points.transpose(), [bestParamsL, bestParamsR], marker_matches


#___________________________________________________________________
#
# Function: listToArray
#			turn a list into an array
# #-------------------------------------------------------------------
# def listToArray(List):
# 	print("List: " + str(List))
# 	rows = len(List)
# 	cols = len(List[0])
# 	print("rows, cols: " + str(rows) + str(cols))

# 	Array = np.zeros((rows,cols))
# 	rowNum = 0
# 	for pt in List: 
# 		Array[rowNum, :] = pt
# 		rowNum = rowNum + 1

# 	print("Array: " + str(Array))
# 	return Array

#___________________________________________________________________

#___________________________________________________________________
#
# Function: saveVideo
#			takes in a sequence of 3D point clouds, saves each points cloud
#			as a frame in the video
# Inputs: 	name: video name 
#			points: list of 3D points
# Return: 	saves a video
# Reference: matlib plot taken from http://matplotlib.org/api/animation_api.html
#-------------------------------------------------------------------
def saveVideo(name, points):
	# set up matlibplot stuff 
	# create figure
	fig = plt.figure()
	# create 3D axes
	ax = Axes3D(fig)
	# counter for what frame the 3D point cloud is from 
	count = len(points)
	
	print("points[0]: " + str(points[0]))
	# set data to the first 3D point cloud
	lines = [ax.scatter(points[0][:, 0], points[0][:, 1], points[0][:, 2], c = 'b', marker = '^')]
	print(type(lines))

	XMIN = np.amin(points[0][:, 0])
	YMIN = np.amin(points[0][:, 1])
	ZMIN = np.amin(points[0][:, 2])
	print("XMIN: " + str(XMIN))

	XMAX = np.amax(points[0][:, 0])
	YMAX = np.amax(points[0][:, 1])
	ZMAX = np.amax(points[0][:, 2])
	print("XMAX: " + str(XMAX))

	boundaries = []
	boundaries.append([XMIN, XMAX])
	boundaries.append([YMIN, YMAX])
	boundaries.append([ZMIN, ZMAX])

	# Setting the axes properties
	ax.set_xlim3d([XMIN, XMAX])
	ax.set_xlabel('X')

	ax.set_ylim3d([YMIN, YMAX])
	ax.set_ylabel('Y')

	ax.set_zlim3d([ZMIN, ZMAX])
	ax.set_zlabel('Z')
	ax.set_autoscale_on(False)

	ax.set_title('SIFT Point Cloud')

	line_ani = animation.FuncAnimation(fig, updatePoints_1PtCloud, count, fargs=(points, lines, fig, boundaries), interval=50, repeat=False)
	
	# save video
	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=15, bitrate=1800)

	line_ani.save(name)

#___________________________________________________________________

#___________________________________________________________________
#
# Function: saveVideoMarkersAndTools
#			Takes in a 3D point clouds of marker and tool points, saves animation of 3D points
#
# Inputs: 	name: video name 
#			marker_points: list of 3D background marker points
#			tool_points: list of 3D tool marker points
# Return: 	saves a video			
# Reference: matlib plot taken from http://matplotlib.org/api/animation_api.html
#-------------------------------------------------------------------
def saveVideoMarkersAndTools(name, marker_points, tool_points):
	# set up matlibplot stuff 
	# create figure
	fig = plt.figure()
	# create 3D axes
	ax = Axes3D(fig)
	# counter for what frame the 3D point cloud is from 
	count = len(marker_points)
	
	# VidName = input('Video name: ')
	# set data to the first 3D point cloud
	lines = [ax.scatter(marker_points[0][:, 0], marker_points[0][:, 1], marker_points[0][:, 2], c = 'b', marker = '^')]
	lines = [ax.scatter(tool_points[0][:, 0], tool_points[0][:, 1], tool_points[0][:, 2], c = 'g', marker = 'o')]
	print(type(lines))

	XMIN_marker = np.amin(marker_points[0][:, 0])
	YMIN_marker = np.amin(marker_points[0][:, 1])
	ZMIN_marker = np.amin(marker_points[0][:, 2])

	XMIN_tool = np.amin(tool_points[0][:, 0])
	YMIN_tool = np.amin(tool_points[0][:, 1])
	ZMIN_tool = np.amin(tool_points[0][:, 2])

	XMIN = np.amin([XMIN_marker, XMIN_tool])
	YMIN = np.amin([YMIN_marker, YMIN_tool])
	ZMIN = np.amin([ZMIN_marker, ZMIN_tool])
	
	XMAX_marker = np.amax(marker_points[0][:, 0])
	YMAX_marker = np.amax(marker_points[0][:, 1])
	ZMAX_marker = np.amax(marker_points[0][:, 2])

	XMAX_tool = np.amax(tool_points[0][:, 0])
	YMAX_tool = np.amax(tool_points[0][:, 1])
	ZMAX_tool = np.amax(tool_points[0][:, 2])

	XMAX = np.amax([XMAX_marker, XMAX_tool])
	YMAX = np.amax([YMAX_marker, YMAX_tool])
	ZMAX = np.amax([ZMAX_marker, ZMAX_tool])
	

	boundaries = []
	boundaries.append([XMIN, XMAX])
	boundaries.append([YMIN, YMAX])
	boundaries.append([ZMIN, ZMAX])

	# Setting the axes properties
	ax.set_xlim3d([XMIN, XMAX])
	ax.set_xlabel('X')

	ax.set_ylim3d([YMIN, YMAX])
	ax.set_ylabel('Y')

	ax.set_zlim3d([ZMIN, ZMAX])
	ax.set_zlabel('Z')
	ax.set_autoscale_on(False)

	ax.set_title('Color Marker Detection')

	line_ani = animation.FuncAnimation(fig, updatePoints, count, fargs=(marker_points, tool_points, lines, fig, boundaries), interval=50, repeat=False)
	
	# save video
	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=30, bitrate=1800)

	line_ani.save(name)
#___________________________________________________________________



#def updatePoints(count, points, data, ax):
def updatePoints(count, marker_points, tool_points, lines, fig, boundaries):
	ax = Axes3D(fig)
	XMIN = boundaries[0][0]
	XMAX = boundaries[0][1]
	YMIN = boundaries[1][0]
	YMAX = boundaries[1][1]
	ZMIN = boundaries[2][0]
	ZMAX = boundaries[2][1]

	# initialize view
	ax.view_init(elev=30, azim=90)

	# Setting the axes properties
	ax.set_xlim3d([XMIN, XMAX])
	ax.set_xlabel('X')

	ax.set_ylim3d([YMIN, YMAX])
	ax.set_ylabel('Y')

	ax.set_zlim3d([ZMIN, ZMAX])
	ax.set_zlabel('Z')
	ax.set_autoscale_on(False)

	ax.set_title('SIFT Point Cloud')
  	# lines = ax.scatter(points[count][:, 0], points[count][:, 1], points[count][:, 2], c=np.random.rand(3,1), marker = '^')
  	lines = [ax.scatter(marker_points[count][:, 0], marker_points[count][:, 1], marker_points[count][:, 2], c = 'b', marker = '^')]
	lines = [ax.scatter(tool_points[count][:, 0], tool_points[count][:, 1], tool_points[count][:, 2], c = 'g', marker = 'o')]
  	print("plotting frame " + str(count))
  	  	
 	return lines

 #def updatePoints(count, points, data, ax):
def updatePoints_1PtCloud(count, points, lines, fig, boundaries):
	ax = Axes3D(fig)
	XMIN = boundaries[0][0]
	XMAX = boundaries[0][1]
	YMIN = boundaries[1][0]
	YMAX = boundaries[1][1]
	ZMIN = boundaries[2][0]
	ZMAX = boundaries[2][1]

	# initialize view
	ax.view_init(elev=45, azim=45)

	# Setting the axes properties
	ax.set_xlim3d([XMIN, XMAX])
	ax.set_xlabel('X')

	ax.set_ylim3d([YMIN, YMAX])
	ax.set_ylabel('Y')

	ax.set_zlim3d([ZMIN, ZMAX])
	ax.set_zlabel('Z')
	ax.set_autoscale_on(False)

	ax.set_title('SIFT Point Cloud')
  	# lines = ax.scatter(points[count][:, 0], points[count][:, 1], points[count][:, 2], c=np.random.rand(3,1), marker = '^')
  	lines = [ax.scatter(points[count][:, 0], points[count][:, 1], points[count][:, 2], c = 'b', marker = '^')]
	
  	print("plotting frame " + str(count))
  	  	
 	return lines

