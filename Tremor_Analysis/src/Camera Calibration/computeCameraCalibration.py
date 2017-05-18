from stereoCalibration import stereoCalibration

#___________________________________________________________________
#
# Function: computeCameraCalibration
# 	    	* read in left and right calibration images 
#			* perform calibration on left/right cameras individually
#			* perform stereo calibration
#			* perform stereo rectification
#			* save results to files
#
# Author:   	Molly O'Brien
# Input: 	leftImsDir: left image directory
#			rightImsDir: right image directory 
#			camCalFile: camera calibration file
#			sterCalFile: stereo calibration file
#			rectCalFile: stereo rectification file 
# Return:	
#------------------------------------------------------------------- 
def computeCameraCalibration(leftImsDir, rightImsDir, camCalFile, sterCalFile, rectCalFile, trialName):
	# Make output file to write F results 
	outputName = trialName + 'StereoCalOutput.txt'
	output = open(outputName,'w')

	# stereo calibration
	objectPoints, leftCalIms, rightCalIms = stereoCalibration(leftImsDir, rightImsDir, camCalFile, sterCalFile, output)

	
