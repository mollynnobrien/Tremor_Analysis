import numpy as np
from dist import dist
import pdb
#__________________________________________________________
#
# Function: greedyMatch
#	
#			given 2 unordered sets of corresponding points, 
#			find matches using greedy search
# 
# \params[in] points1: first point cloud
# \params[in] points2: second point cloud
# \return: ordered_points1: points in first point cloud orders so ordered_points1[i] matches with ordered_points2[i]
# \return: ordered_points2: points in second point cloud orders so ordered_points1[i] matches with ordered_points2[i]
#-----------------------------------------------------------
def greedyMatch(points1, points2):
	len1 = len(points1)
	len2 = len(points2) 
	# get number of points 
	numPoints = np.amin((len1, len2))
	# form distance matrix
	D = np.zeros((len1, len2))

	# add distances
	for row in range(len1):
		for col in range(len2):
			D[row, col] = dist(points1[row], points2[col])

	# find matches 
	ordered_pts1 = []
	ordered_pts2 = []

	for it in range(numPoints):
		min_dist = np.amin(D)
		# get indices of min_dist
		min_row, min_col = np.where(D == min_dist)
		# points1[min_row] matches points2[min_col]
		ordered_pts1.append(points1[min_row])
		ordered_pts2.append(points2[min_col])

		# set D along (min_row, min_col) to a really high number
		D[min_row, :] = 1000000000*np.ones((1, len2))
		D[:, min_col] = 1000000000*np.ones((len1, 1))

	# return matched points 
	return ordered_pts1, ordered_pts2
