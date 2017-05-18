import cv2 
import sys
import os
import pdb
import pickle

sys.path.insert(0, 'colorMarkerDetection/')

from markerHistogramExtractor import computeMarkerProb
#_______________________________________________________
# 
# Function: createMarkerHistFile
# 
#			let the user pick which probability image channel gives good color 
#			marker detections
# 
# \params[in] Folder_Name: experiment folder name
# \params[in] background_Hist: color histogram for background markers
# \params[in] tool_Hist: color histogram for tool markers
# \params[in] N_m:  number of Gaussian mixture model modes for the background marker probability
# \params[in] N_t: number of Gaussian mixture model modes for the tool marker probability
# \return N_m_select: channel of background probability image with background marker detections
# \return N_t_select:  channel of tool probability image with tool marker detections
#--------------------------------------------------------
def createMarkerHistFile(Folder_Name, background_Hist, tool_Hist, N_m, N_t):
	# open an image with background markers
	for im in os.listdir('experiments/' + Folder_Name + '/Color_Markers/Background/images/'):
		back_im = cv2.imread('experiments/' + Folder_Name + '/Color_Markers/Background/images/' + im, 1)
		break

	pdb.set_trace()
	# apply background hist to a background image 
	probImBackgroundAll = computeMarkerProb(back_im, background_Hist, N_m, 60)
	# display each channel of the probability image
	for channel in range(N_m):
		cv2.imshow("Background channel " + str(channel), probImBackgroundAll[channel])
		cv2.waitKey()

	# which channel looks good? 
	N_m_select = input("Which channel should be used for the background prob im? \t")
	cv2.destroyAllWindows()

	# open an image with tool markers
	for im in os.listdir('experiments/' + Folder_Name + '/Color_Markers/Tool/images/'):
		tool_im = cv2.imread('experiments/' + Folder_Name + '/Color_Markers/Tool/images/' + im, 1)
		break
	# apply background hist to a tool image 
	probImToolAll = computeMarkerProb(back_im, tool_Hist, N_t, 20)
	# display each channel of the probability image
	for channel in range(N_t):
		cv2.imshow("Tool channel " + str(channel), probImToolAll[channel])
		cv2.waitKey()

	# which channel looks good? 
	N_t_select = input("Which channel should be used for the tool prob im? \t")
	cv2.destroyAllWindows()

	histInfo = [N_m_select, N_t_select]
	
	# save file
	fileName = "experiments/" + Folder_Name + "/maker-histogram-info.txt"
	with open(fileName, 'wb') as f:
		pickle.dump(histInfo, f)

	return N_m_select, N_t_select

