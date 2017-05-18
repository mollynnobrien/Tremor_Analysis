import sys
sys.path.insert(0, 'src/')
from Tremor_Analysis import Tremor_Analysis

#____________________________________________________________
#
# Function: runTremorAnalysis
#
#			Fill in video names and color marker information,
#			get hand tremor information 
#
# Author: 	Molly O'Brien 04/27/2017
#
# TO USE: 	put all files in a folder named Folder_Name
#
#			In the folder:
#				Calibration/Left_Ims/*
#					> Left calibration images 
#					> image names should be the frame number
#				Calibration/Right_Ims/*
#					> Right calibration images saved in 
#					> image names should be the frame number
#
#				left_video.avi
#					> left camera experimental video 
#				right_video.avi
#					> right camera experimental video 
#
#				Color_Markers/Background/images
#					> images with color markers
# 					> image names should be the frame number
#				Color_Markers/Background/binary_masks
#					> masks around marker
# 					> image names should be the frame number
#				Color_Markers/Background/template.png
#					> template image
#				Color_Markers/Tool/images
#					> images with tool markers showing
# 					> image names should be the frame number
#				Color_Markers/Tool/binary_masks
#					> masks around markers
# 					> image names should be the frame number
#
# TO CALL FUNCTION: python runTremorAnalysis [Folder_Name]
#-------------------------------------------------------------

def runTremorAnalysis():
	#---------------------------------------------------------
	# 	===== Check Usage =====
	if(len(sys.argv) < 2):
			print("Error! \n Usage: python runTremorAnalysis.py [folder name]")
			return
	#---------------------------------------------------------
	# 	===== Folder Name =====
	Folder_Name = sys.argv[1] 

	#---------------------------------------------------------
	# 	===== Call tremor analysis ======
	Tremor_Analysis(Folder_Name)

#______________________________________________________________

runTremorAnalysis()