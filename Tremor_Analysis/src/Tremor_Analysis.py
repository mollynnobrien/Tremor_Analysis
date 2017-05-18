import os
import sys 
import cv2
import pdb
import pickle
# add folders to path
sys.path.insert(0, 'src/Color Marker Detection/')
sys.path.insert(0, 'src/Bundle Adjustment/')
sys.path.insert(0, 'src/Camera Calibration/')
sys.path.insert(0, 'src/Frequency Analysis/')

# import files
from computeCameraMovement import computeCameraMovement
from computeCameraCalibration import computeCameraCalibration
from blobDetector import MSERBlobDetector, MSERBlobDetectorDebugging
from markerHistogramExtractor import markerHistogramExtractor, computeGaussMixModel
from createMarkerHistFile import createMarkerHistFile
from bundleAdjustment import apply_bundle_adjustment
from ICP import stabilizePoints
from buildTrajectory import runFrequencyAnalysis
from blobParamsApp import createBlobDetectorFile
from greedyMatch import greedyMatch
#_____________________________________________________________
#
# Function: Tremor_Analysis
#
#			Performs frequency analysis on tool movement in 
#			surgical experiments
#
# \params[in] Folder_Name: 	folder with videos, calibration images, and 
#							marker information 
#--------------------------------------------------------------
def Tremor_Analysis(Folder_Name):
	#========================================================================
	#
	#	Set-Up 
	#
	#========================================================================
	#-----------------------------------------------------------
	#		===== Calibration =====

	leftImsDir = 'experiments/' + Folder_Name + '/Calibration/Left_Ims/'
	rightImsDir = 'experiments/' + Folder_Name + '/Calibration/Right_Ims/'

	# calibration file names 
	camCalFile = 'experiments/' + Folder_Name + "/camera-calibration.txt"
	sterCalFile = 'experiments/' + Folder_Name + "/stereo-calibration.txt"
	rectCalFile = 'experiments/' + Folder_Name + "/rectify-calibration.txt"

	# if the files aren't there already
	if(not(os.path.isfile(camCalFile) and os.path.isfile(sterCalFile))):
		# perform the calibration 
		computeCameraCalibration(leftImsDir, rightImsDir, camCalFile, sterCalFile, rectCalFile, Folder_Name)

	#------------------------------------------------------------
	# 		===== Color Marker Information =====

	# background marker folders
	background_image_dir = 'experiments/' + Folder_Name + '/Color_Markers/Background/images/'
	background_mask_dir = 'experiments/' + Folder_Name + '/Color_Markers/Background/binary_masks/'
	# background template image 
	background_template_L = cv2.imread('experiments/' + Folder_Name + '/Color_Markers/Background/template_left.png', 0)
	# background template image 
	background_template_R = cv2.imread('experiments/' + Folder_Name + '/Color_Markers/Background/template_right.png', 0)
	# find center of marker keypoints 
	keypoints, template_points_L, blobImage = MSERBlobDetectorDebugging(background_template_L, "background_template")
	# find center of marker keypoints 
	keypoints, template_points_R, blobImage = MSERBlobDetectorDebugging(background_template_R, "background_template")

	# get ordered points so pointL[i] corresponds to pointsR[i]
	template_points_L, template_points_R = greedyMatch(template_points_L, template_points_R)

	# tool marker folders
	tool_image_dir = 'experiments/' + Folder_Name + '/Color_Markers/Tool/images/'
	tool_mask_dir = 'experiments/' + Folder_Name + '/Color_Markers/Tool/binary_masks/'

	# file for marker histogram information 
	markerHistFile = 'experiments/' + Folder_Name + '/maker-histogram-info.txt'
	# file for tool marker area info
	blobFile = 'experiments/' + Folder_Name + '/tool-blob-info.txt'
	# flag to see if we need to find marker histogram info
	needMarkerHistInfo = True
	# flag to see if we need to find tool blob size
	needBlobFile = True
	# check to see if histogram info already saved
	if(os.path.isfile(markerHistFile)):
		# we don't need to make the histogram info file 
		needMarkerHistInfo = False
		# open marker histogram file 
		with open(markerHistFile, 'rb') as f:
			markerHistResults = pickle.load(f)
			# load the correct marker histogram prob channel
			N_m_select = markerHistResults[0]
			N_t_select = markerHistResults[1]

	N_m = 3
	N_t = 4


	if(os.path.isfile(blobFile)):
		# we don't need to make the histogram info file 
		needBlobFile = False
		# open marker histogram file 
		with open(blobFile, 'rb') as f:
			blobResults = pickle.load(f)
			# load the correct marker histogram prob channel
			tool_min_area = blobResults[0]
			tool_max_area = blobResults[1]

	
	#................................................................
	# get histogram of background marker colors
	marker_HS_Samples 	= markerHistogramExtractor(background_image_dir, background_mask_dir)
	# make Gaussian mixture model
	marker_HS_em = computeGaussMixModel(N_m, marker_HS_Samples)#, output)

	#................................................................
	# get histogram of tool marker colors
	tool_HS_Samples 	= markerHistogramExtractor(tool_image_dir, tool_mask_dir)
	# make Gaussian mixture model
	tool_HS_em = computeGaussMixModel(N_t, tool_HS_Samples)#, output)

	#................................................................
	if(needMarkerHistInfo):
		N_m_select, N_t_select = createMarkerHistFile(Folder_Name, marker_HS_em,  tool_HS_em, N_m, N_t)
	if(needBlobFile):
		tool_min_area, tool_max_area = createBlobDetectorFile(Folder_Name, tool_HS_em, N_t_select)

	# save marker histogram info
	marker_HistInfo = [marker_HS_em, N_m, N_m_select, "background"]
	# save tool histogram info
	tool_HistInfo = [tool_HS_em, N_t, N_t_select, "tool", tool_min_area, tool_max_area]
	
	#-----------------------------------------------------------
	# save all desired detector histograms in one list
	HistInfo = [tool_HistInfo, marker_HistInfo]

	#----------------------------------------------------------------------
	# 			===== OPEN VIDEOS =====
	capLeft = cv2.VideoCapture('experiments/' + Folder_Name + "/left_video.avi")
	capRight = cv2.VideoCapture('experiments/' + Folder_Name + "/right_video.avi")

	# get size from videos
	List = capLeft.read()
	List = capRight.read()

	#----------------------------------------------------------
	# 		===== Outputs =====

	# set up output file
	outputName = Folder_Name + 'ComputeCameraMovementOutput.txt'
	outputFile = open(outputName,'w')

	# open output video to show marker detections to
	outL = cv2.VideoWriter(Folder_Name + '_markerDetectionL.avi',cv2.VideoWriter_fourcc(*'MJPG'),32,(List[1].shape[1],List[1].shape[0]))
	# open output video to show marker detections to
	outR = cv2.VideoWriter(Folder_Name + '_markerDetectionR.avi',cv2.VideoWriter_fourcc(*'MJPG'),32,(List[1].shape[1],List[1].shape[0]))

	#========================================================================
	#
	#	Computation 
	#
	#========================================================================
	# --------------------------------------------------------------
	#    ====== Call computeCameraMovement =====
	marker_points3D, tool_points3D, T = computeCameraMovement(capLeft, capRight, HistInfo, camCalFile, sterCalFile, outputFile, Folder_Name, outL, outR, background_template_L, template_points_L, background_template_R, template_points_R)

	#------------------------------------------------------------------------
	#			===== Bundle Adjustment =====
	T_optimal = apply_bundle_adjustment(marker_points3D, T)
	# write save transforms to output file
	with open(Folder_Name+"Tand3Dpoints_bundleAdjustment.txt", 'wb') as f:
		pickle.dump([marker_points3D, tool_points3D, T_optimal], f)
	# # stabilize marker points 
	marker_points_stable = stabilizePoints(marker_points3D, T_optimal, outputFile)
	pdb.set_trace()
	tool_points_stable = stabilizePoints(tool_points3D, T_optimal, outputFile)

	#-------------------------------------------------------------------------
	#		===== Frequency Analysis =====
	runFrequencyAnalysis(tool_points_stable)


	#==========================================================================
	#==========================================================================