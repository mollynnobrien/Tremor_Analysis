Tremor Analysis Code 
Molly O'Brien 		5/18/17
mobrie38@jhu.edu
#----------------------------------------------------------------------------

TO USE: 	put all files in Tremor_Analysis/experiments/Folder_Name

			In the folder:
				Calibration/Left_Ims/*
					> Left calibration images 
					> image names should be the frame number
				Calibration/Right_Ims/*
					> Right calibration images saved in 
					> image names should be the frame number
				left_video.avi
					> left camera experimental video 
				right_video.avi
					> right camera experimental video 

				Color_Markers/Background/images
					> images with color markers
 					> image names should be the frame number
				Color_Markers/Background/binary_masks
					> masks around marker
 					> image names should be the frame number
				Color_Markers/Background/template.png
					> template image
				Color_Markers/Tool/images
					> images with tool markers showing
 					> image names should be the frame number
				Color_Markers/Tool/binary_masks
					> masks around markers
 					> image names should be the frame number

* experiments/sample_folder is an empty folder with the correct folder structure needed for each experiment

# TO CALL FUNCTION: 
cd “path to folder”/Tremor_Analysis/
python runTremorAnalysis [Folder_Name]

