import cv2
import numpy as np
import pickle
import pdb
import matplotlib.pyplot as plt
import numpy.linalg as LA
import math
import sys

from dist import dist

sys.path.insert(0, '../')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#_______________________________________________________________
#
# Function: buildTrajectory
#
#			take in cloud of 3D points for video frame, 
#			associate points across frames to generate trajectories 
#
# \params[in]: points 		list of 3D point clouds in video frames
# \return: trajectory 		list of position trajectories 
#----------------------------------------------------------------
def buildTrajectory(points):
	# list of all found trajectories
	trajectory = []
	# how close 2 points need to be to be linked 
	PT_THRESH = 1
	# find number of frames 
	numFrames = len(points)
	
	#------------------------------------------
	# set up prevPoints as first frame 
	prevPoints = points[0]
	# number of points in previous frame
	numPrev = len(prevPoints)

	# start a trajectory for each point in the first frame 
	for idx in range(len(prevPoints)):
		# get pt
		pt = prevPoints[idx]
		# start trajectory with pt
		trajectory.append([pt])
		
	#--------------------------------------------
	# go through the 3D point cloud generated for each video frame
	for idx in range(1, numFrames): 
		# get next frame of points
		curPoints = points[idx]	
		# find number of current points 
		numCur = len(curPoints)	
		# for each 3D point triangulated in the video frame
		for idx_c in range(numCur):
			# point match hasn't been found yet
			foundFlag = False
			# get point
			curPt = curPoints[idx_c]
			# is this very close to a pt in an existing trajectory? 
			for trajNum in range(len(trajectory)):
				# get last point in this trajectory
				prevPt = trajectory[trajNum][-1]
				# if yes-> they are part of the same trajectory
				displacement = dist(curPt, prevPt)

				if(displacement < PT_THRESH):
					# add curPt to trajNum trajectory
					trajectory[trajNum].append(curPt)
					# remove curPt so we know which pts aren't matched
					np.delete(curPoints, (idx_c), axis=0)
					# switch flag saying point was found 
					foundFlag  = True
					break

			# if didn't find a match in an existing traj, check missed points 
			if(not foundFlag):
				# go through all missed points
				for missedPt in missedPoints:
					# find distance btwn missed pt and curPt
					distance = dist(curPt, missedPt)
					# if distnace small
					if (distance < PT_THRESH):
						# create new trajectory
						trajNum = len(trajectory)
						# start new trajectory with both points
						trajectory.append([missedPt, curPt])
						# remove curPt so we know which pts aren't matched
						np.delete(curPoints, (idx_c), axis=0)
						# switch flag saying point was found 
						foundFlag  = True
							
		#----------------------------------
		# update for next frame 
		# points still in curPoints are the missed points 
		missedPoints = curPoints
		
	#-------------------------------------------------------
	return trajectory

#______________________________________________________
#
# Function: pruneTraj
#
#			only save trajectories with len> MIN_LEN
# 
# \params[in] traj: list of trajectories
# \return longTraj: list of long trajectories
#-------------------------------------------------------
def pruneTraj(traj):
	# shortest valid trajectory, need this many points to be a valid trajectory
	MIN_TRAJ_LEN = 30
	# declare variable for long trajs
	longTraj = []
	# declare flat list (all disp in 1 list)
	flatDisp = []

	for (path) in zip(traj): 
		# pdb.set_trace()
		if(len(path[0]) >= MIN_TRAJ_LEN):			
			longTraj.append(path)

	return longTraj


#______________________________________________________
# 
# Function: build xyz trajectories 
# 
# 			take trajectories of 3D points and seperate into list of x, y, 
#			and z coordinates
#
# \params[in] trajectory: trajectory of 3D points
# \return x_traj: trajectory of x coordinates
# \return y_traj: trajectory of y coordinates
# \return z_traj: trajectory of z coordinates
#-------------------------------------------------------
def buildXYZ_Trajectories(trajectory):
	# define x, y, z trajectories 
	x_traj = []
	y_traj = []
	z_traj = []

	# go through points on trajectory
	for pt in trajectory[0]: 
		# pdb.set_trace()
		# add x_coord to x_traj
		x_traj.append(pt[0])
		# add y_coord to y_traj
		y_traj.append(pt[1])
		# add z_coord to z_traj
		z_traj.append(pt[2])

	# return list of x, y, z coordinates
	return x_traj, y_traj, z_traj
#_______________________________________________________
# 
# Function: linkToolTrajs(trajectories)
# 
# 			given short, absolute trajectories, link into 1 
# 			continuous trajectory by subtracting the translational offset 
# 			between traj_n-1[last] and traj_n[first]
#
# \params[in] trajectories: list of absolute trajectories
# \return linked_traj: long trajectory with relative positions from all trajectories
#---------------------------------------------------------
def linkToolTrajs(trajectories):
	# declare long trajectory
	linked_traj = []
	# add the first trajectory
	linked_traj.extend(trajectories[0][0])
	# iterate through individual trajectories 
	for idx in range(1, len(trajectories)):
		traj = trajectories[idx][0]
		# offset is difference between first element in traj and 
		# last element in linked_traj
		offset = traj[0] - linked_traj[-1]
		# add traj-offset to linked_traj
		for pt in traj:
			linked_traj.extend([pt - offset])
	# return the linked trajectory
	return linked_traj

#________________________________________________________
#
# Function: rotatePoints
# 
# 			given a list of 3D coordinates, transform by given 4x4 homogeneous 
#			transform 
# 
# \param[in] points: 	Nx3 array of points 
# \param[in] T:			4x4 homogeneous transform 
# \return points_rot: 	points transformed by T
#---------------------------------------------------------
def rotatePoints(points, T): 
	#print("in rotate points")
	pt = 0
	worldPoints = np.zeros((3,len(points)))
	# extract rotation 
	R = T[0:3, 0:3]
	# extract translation 
	t = T[0:3, 3:4]
	for point in points:
		# print("point: " + str(point))
		# make point a column vector
		point.shape = (3,1)
		# [R_world_cam|t]*corners_world
		worldPoint = np.dot(R, point) + t
		# make cameraPoint a column vector
		worldPoint.shape = (3,1)
		# save cameraPoint in matrix cameraPoints
		worldPoints[0:3,pt:pt+1] = worldPoint
		# increment pt
		pt = pt + 1

	worldPoints = worldPoints.transpose()
	
	return worldPoints
#_______________________________________________________
# 
# Function: frequncyAnalysis
#
# 			given a function and a sampling frequency, take the DFT, convert
#			to angle and magnitude representation, high pass filter to see tremor, 
#			compare 3D trajectory with and without tremor. 
#
# \params[in] function: signal to take DFT of
# \params[in] FsBy2: sampling frequency / 2
# \return 	f_tremor: the DFT magnitude of function in tremor range
#-------------------------------------------------------
def frequencyAnalysis(function, FsBy2):
	# number of tool samples
	N = len(function[0]) 

	# get x, y, z components

	# reshape to get arrays of x, y, z displacements
	func_x_Flat = np.reshape(function[0], (1, N))
	func_y_Flat = np.reshape(function[1], (1, N))
	func_z_Flat = np.reshape(function[2], (1, N))

	# take dft in each direction
	fx = cv2.dft(func_x_Flat)/N	
	fy = cv2.dft(func_y_Flat)/N
	fz = cv2.dft(func_z_Flat)/N



	if np.mod(N, 2) == 0: 
		print("even")
		# list of real indices in DFT
		real_indices = [0]
		real_indices.extend(range(1, N, 2))
		# list of imaginary indices in DFT
		imag_indices = [range(2, N-1, 2)]
		# extract real and imaginary x-components
		fx_real = [fx[0][ri] for ri in real_indices]
		fy_real = [fy[0][ri] for ri in real_indices]
		fz_real = [fz[0][ri] for ri in real_indices]
		# add 0 at DC 
		fx_imag = [0]
		fy_imag = [0]
		fz_imag = [0]
		for ii in imag_indices:
			fx_imag.extend(fx[0][ii])
			fy_imag.extend(fy[0][ii])
			fz_imag.extend(fz[0][ii])
		# there is one more real comp than imag, so add 0 at end
		fx_imag.extend([0])
		fy_imag.extend([0])
		fz_imag.extend([0])
		# pdb.set_trace()


	else:
		print("odd")
		# list of real indices in DFT
		real_indices = [0]
		real_indices.extend(range(1, N -2, 2))
		# list of imaginary indices in DFT
		imag_indices = [range(2, N-1, 2)]
		# extract real and imaginary x-components
		fx_real = [fx[0][ri] for ri in real_indices]
		fy_real = [fy[0][ri] for ri in real_indices]
		fz_real = [fz[0][ri] for ri in real_indices]
		# add 0 at DC 
		fx_imag = [0]
		fy_imag = [0]
		fz_imag = [0]
		for ii in imag_indices:
			fx_imag.extend(fx[0][ii])
			fy_imag.extend(fy[0][ii])
			fz_imag.extend(fz[0][ii])
		
	# convert to arrays 
	fx_real = np.reshape(fx_real, (1, len(fx_real)))
	fx_imag = np.reshape(fx_imag, (1, len(fx_imag)))
	# convert to arrays 
	fy_real = np.reshape(fy_real, (1, len(fy_real)))
	fy_imag = np.reshape(fy_imag, (1, len(fy_imag)))
	# convert to arrays 
	fz_real = np.reshape(fz_real, (1, len(fz_real)))
	fz_imag = np.reshape(fz_imag, (1, len(fz_imag)))

	# convert from real, imaginary representation to magnitude, angle representation
	fx_mag, fx_ang = cv2.cartToPolar(fx_real[0], fx_imag[0])
	fy_mag, fy_ang = cv2.cartToPolar(fy_real[0], fy_imag[0])
	fz_mag, fz_ang = cv2.cartToPolar(fz_real[0], fz_imag[0])
	
	# get number of points in real/image
	Nby2 = len(fx_real[0])
	# make list 1:N_tool
	list_freq = np.float32(range(Nby2))
	# convert to Hz
	w = [ l*FsBy2/Nby2 for l in list_freq]

	# pdb.set_trace()
	# print("showing fx_mag")
	# plt.stem(w, fx_mag)
	# plt.show()
	
	# # pdb.set_trace()
	# print("showing fy_mag")
	# plt.stem(w, fy_mag)
	# plt.show()
	
	# print("showing fz_mag")
	# plt.stem(w, fz_mag)
	# plt.show()

	# get sum of tremor at each frequency
	f_mag = []
	for w_i in range(len(w)): 
		f_sum = fx_mag[w_i] + fy_mag[w_i] + fz_mag[w_i]
		f_mag.extend(f_sum)

	# show f_mag
	plt.stem(w, f_mag)
	plt.show()
	# filter f_mag to just show frequency compenents in tremor range
	f_tremor = filterTremor(w, f_mag)
	
	# DFT of all motion
	f_all = [fx, fy, fz]
	comparePaths(f_all, 5, w)

	return f_tremor
#_______________________________________________________

#_______________________________________________________
#
# Function: filterTremor
#
# 			given trajectory frequency analysis results, find the 
#			frequency components magnitude in the tremor region
# 
# \params[in] w: array with frequency corresponding to each element in f (Hz)
# \params[in] f: frequency component magnitudes
# \return f: 	f filtered so components above tremor threshold are 0
# \return w: 	w (same as before) 
#-------------------------------------------------------
def filterTremor(w, f): 
	# define tremor frequency in Hz
	tremor_minFreq = 5
	tremo_maxFreq = 15
	# get mask for frequency components in tremor range
	tremor_bool = [ (wi > tremor_minFreq and wi < tremo_maxFreq) for wi in w]
	# multiply fx, fy, fz by tremor_bool
	for idx in range(len(w)):
		f[idx] = f[idx]*tremor_bool[idx]

	# show plot
	# plt.stem(w, f)
	# plt.show()

	# find the amplitude and frequency of largest tremor component
	max_amp = np.amax(f)	
	max_tremor_freq = w[f.index(max_amp)]	
	
	return f, w
#_______________________________________________________

#________________________________________________________
#
# Function: comparePaths
#
#			given a dft and a tremor threshold plot the path with
#			tremor and the path without
# 
# \params[in] f: frequency component magnitudes
# \params[in] tremor_thresh: tremor threshold (Hz)
# \params[in] w: array with frequency corresponding to each element in f (Hz)
# \return: plot 3D trajectories
#--------------------------------------------------------
def comparePaths(f, tremor_thresh, w):
	fx = f[0]
	fy = f[1]
	fz = f[2]
	
	# get binary mask for intentional motion 
	counter = 0
	# get max index that is intentional motion in extended notation 
	for w_i in w: 
		if w_i < tremor_thresh: 
			w_index_max = counter 
		counter = counter + 1
	print("w_index_max: " + str(w_index_max))
	print("w[w_index_max]: " + str(w[w_index_max]))
	#  extended flag for when real & imaginary components together
	w_ext_index_max = 2*w_index_max - 1
	# declare smooth arrays
	fx_smooth = np.zeros(fx.shape)
	fy_smooth = np.zeros(fx.shape)
	fz_smooth = np.zeros(fx.shape)
	
	# create f_smooth
	for idx in range(len(fx[0])):
		fx_i = fx[0][idx]
		fy_i = fy[0][idx]
		fz_i = fz[0][idx]
		if(idx <= w_ext_index_max):
			fx_smooth[0][idx] = fx_i
			fy_smooth[0][idx] = fy_i
			fz_smooth[0][idx] = fz_i
			print("idx added to smooth: " + str(idx))

	# take idft of f
	x_all = cv2.idft(fx)
	y_all = cv2.idft(fy)
	z_all = cv2.idft(fz)
	
	# take idft of f_smooth
	x_smooth = cv2.idft(fx_smooth)
	y_smooth = cv2.idft(fy_smooth)
	z_smooth = cv2.idft(fz_smooth)

	# convert DFT into magnitude, angle representation
	fx_smooth_mag, fx_smooth_ang, wi = convertDFT(fx_smooth, len(fx_smooth[0]), 15)
	# plot results
	plt.stem(w, fx_smooth_mag)
	plt.show()
	
	# plot the 2 paths 
	fig = plt.figure()
	ax = Axes3D(fig)
	lines = [ax.plot(x_smooth[0], y_smooth[0], c="k",  zs=z_smooth[0])]
	lines = [ax.plot(x_all[0], y_all[0],  c="r", zs=z_all[0])]
	plt.show()


	
#________________________________________________________
# 
# Function: inverseToPath
#
#			given a DFT, find the path in world coordinates 
# \params[in] fx: DFT(x(t))
# \params[in] fy: DFT(y(t))
# \params[in] fz: DFT(z(t))
# \return path: 3D path in time
#--------------------------------------------------------
def inverseToPath(fx, fy, fz):
	# take inverse DFT of x trajectory 
	idft_x = cv2.idft(fx)
	# take inverse DFT of y trajectory 
	idft_y = cv2.idft(fy)
	# take inverse DFT of z trajectory 
	idft_z = cv2.idft(fz)

	# form into a path
	path = []
	for(x, y, z) in zip(idft_x, idft_y, idft_z):
		# save x,y,z coords for each step
		path.append([x, y, z])

	# return path
	return path

#_______________________________________________________

#_______________________________________________________
#
# Function: convertDFT
# 
#			convert DFT from cv2 format to mag and angle
#-------------------------------------------------------
def convertDFT(f, N, FsBy2):
	if np.mod(N, 2) == 0: 
		# EVEN N
		# list of real indices in DFT
		real_indices = [0]
		real_indices.extend(range(1, N, 2))
		# list of imaginary indices in DFT
		imag_indices = [range(2, N-1, 2)]
		# extract real and imaginary x-components
		f_real = [f[0][ri] for ri in real_indices]
		
		# add 0 at DC 
		f_imag = [0]
		# gather imaginary components
		for ii in imag_indices:
			f_imag.extend(f[0][ii])
			
		# there is one more real comp than imag, so add 0 at end
		f_imag.extend([0])
		
	else:
		# ODD N
		# list of real indices in DFT
		real_indices = [0]
		real_indices.extend(range(1, N -2, 2))
		# list of imaginary indices in DFT
		imag_indices = [range(2, N-1, 2)]
		# extract real and imaginary x-components
		f_real = [f[0][ri] for ri in real_indices]
		# add 0 at DC 
		f_imag = [0]
		
		for ii in imag_indices:
			f_imag.extend(f[0][ii])
			
	# convert to arrays 
	f_real = np.reshape(f_real, (1, len(f_real)))
	f_imag = np.reshape(f_imag, (1, len(f_imag)))

	# convert from real, imaginary to magnitude, angle representation
	f_mag, f_ang = cv2.cartToPolar(f_real[0], f_imag[0])
	
	# get number of points in real/image
	Nby2 = len(f_real[0])
	# make list 1:N_tool
	list_freq = np.float32(range(Nby2))
	# convert to Hz
	w = [ l*FsBy2/Nby2 for l in list_freq]

	return f_mag, f_ang, w
#_______________________________________________________________


#_______________________________________________________
#
# Function: runFrequencyAnalysis
# 
#			call this function to run the frequency analysis on a 
#			set of 3D points
#--------------------------------------------------------
def runFrequencyAnalysis(tool_points3D):
	Fs = 30 #fps

	# link nearby points into trajectories
	traj = buildTrajectory(tool_points3D)
	# remove trajectories < 4 pts long
	tool_X = pruneTraj(traj)
	# get one tool trajectory 
	# tool_X_linked = linkToolTrajs(tool_X)
	# get x, y, z trajectories
	tool_x_linked, tool_y_linked, tool_z_linked = buildXYZ_Trajectories(traj)
	# frequency analysis
	frequencyAnalysis([tool_x_linked, tool_y_linked, tool_z_linked], Fs/2)



#____________________________________________________________________
#
# Function: compareRobotAndManual
#	
# 			plot hand-held and robot-held tool tracking results side by side
#----------------------------------------------------------------------
def compareRobotAndManual():
	# robot trial name
	robotTrial = '../suturing_robot_05-04-17-1'
	# open 3D points
	with open(robotTrial+"Tand3Dpoints_save.txt", 'rb') as f:
		robotResults = pickle.load(f)

	# manual trial name
	manualTrial = '../suturing_manual_05-04-17-1'
	# open 3D points
	with open(manualTrial+"Tand3Dpoints.txt", 'rb') as f:
		manualResults = pickle.load(f)

	robotToolPoints = robotResults[1]
	manualToolPoints = manualResults[1]

	Fs = 30

	for frame in range(len(robotToolPoints)):
		# take transpose of points 
		robotToolPoints[frame] = robotToolPoints[frame].transpose()

	for frame in range(len(manualToolPoints)):
		# take transpose of points 
		manualToolPoints[frame] = manualToolPoints[frame].transpose()

	#--------------------------------------------
	# 	====== Robot Frequency Analysis =====
	# link nearby points into trajectories
	roboTraj = buildTrajectory(robotToolPoints)
	# remove trajectories < 4 pts long
	robot_X = pruneTraj(roboTraj)
	# get one tool trajectory 
	# robot_X_linked = linkToolTrajs(robot_X)

	# find longest single trajectory 
	max_len = 0 
	max_idx = 0
	idx = 0
	for traj in robot_X: 
		cur_len = len(traj[0])
		if(cur_len > max_len):
			max_len = cur_len
			max_idx = idx

		idx = idx + 1
	print("Max robot len: " + str(max_len))

	# get x, y, z trajectories
	# robot_x_linked, robot_y_linked, robot_z_linked = buildXYZ_Trajectories([robot_X_linked])
	# frequency analysis
	robot_x_linked, robot_y_linked, robot_z_linked = buildXYZ_Trajectories(robot_X[max_idx])
	robot_f_tremor, w_rob = frequencyAnalysis([robot_x_linked, robot_y_linked, robot_z_linked], Fs/2)


	#--------------------------------------------
	# 	====== Manual Frequency Analysis =====
	# link nearby points into trajectories
	manTraj = buildTrajectory(manualToolPoints)
	# remove trajectories < 4 pts long
	man_X = pruneTraj(manTraj)
	# get one tool trajectory 
	# man_X_linked = linkToolTrajs(man_X)
	max_len = 0 
	max_idx = 0
	idx = 0
	for traj in man_X: 
		cur_len = len(traj[0])
		if(cur_len > max_len):
			max_len = cur_len
			max_idx = idx

		idx = idx + 1

	print("Max man len: " + str(max_len))

	# get x, y, z trajectories
	# man_x_linked, man_y_linked, man_z_linked = buildXYZ_Trajectories([man_X_linked])
	man_x_linked, man_y_linked, man_z_linked = buildXYZ_Trajectories(man_X[max_idx])
	# frequency analysis
	man_f_tremor, w_man = frequencyAnalysis([man_x_linked, man_y_linked, man_z_linked], Fs/2)

	#----------------------------------------------
	#	===== Plot Results =====
	f, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
	ax1.stem(w_rob, robot_f_tremor)
	ax1.set_title('Robot Tremor')
	ax2.stem(w_man, man_f_tremor)
	ax2.set_title('Manual Tremor')

	
	plt.show()

#____________________________________________________________________
#
# Function: compareRobotAndManual
#	
# 			plot tracking results with and without camera motion compensation side by side
#----------------------------------------------------------------------
def beforeAfterMotionComp():
	# robot trial name
	robotTrial = '../suturing_robot_05-04-17-1'
	# open 3D points
	with open(robotTrial+"Tand3Dpoints_save.txt", 'rb') as f:
		robotResults = pickle.load(f)

	# get bundle adjustment results
	with open(robotTrial+"Tand3Dpoints_bundleAdjustment.txt", 'rb') as f:
		bundleAdjResults = pickle.load(f)

	markerPoints = robotResults[0]
	toolPoints = robotResults[1]
	
	T_optimal = bundleAdjResults[2]

	Fs = 30

	marker_stable = []
	tool_stable = []

	for frame in range(len(toolPoints)):
		# take transpose of points 
		toolPoints[frame] = toolPoints[frame].transpose()

	for frame in range(0,len(T_optimal)): 
		m_pts = markerPoints[frame]
		m_pts_stable = rotatePoints(m_pts, T_optimal[frame])
		marker_stable.append(m_pts_stable)

		t_pts = toolPoints[frame]
		t_pts_stable = rotatePoints(t_pts, T_optimal[frame])
		tool_stable.append(t_pts_stable)


	markerFlag = False
	if(markerFlag):
		#--------------------------------------------
		# 	====== Marker Frequency Analysis =====
		# link nearby points into trajectories
		markerTraj = buildTrajectory(markerPoints)
		# remove trajectories < 4 pts long
		marker_X = pruneTraj(markerTraj)
		# get one tool trajectory 
		# robot_X_linked = linkToolTrajs(robot_X)
		# get x, y, z trajectories
		marker_x, marker_y, marker_z = buildXYZ_Trajectories(marker_X[0])
		# frequency analysis
		marker_f_tremor, w_marker = frequencyAnalysis([marker_x, marker_y, marker_z], Fs/2)


		#--------------------------------------------
		# 	====== Stabilized Frequency Analysis =====
		# link nearby points into trajectories
		stabTraj = buildTrajectory(marker_stable)
		# remove trajectories < 4 pts long
		stab_X = pruneTraj(stabTraj)
		# get one tool trajectory 
		# stab_X_linked = linkToolTrajs(stab_X)
		# get x, y, z trajectories
		stab_x, stab_y, stab_z = buildXYZ_Trajectories(stab_X[0])
		# frequency analysis
		stab_f_tremor, w_stab = frequencyAnalysis([stab_x, stab_y, stab_z], Fs/2)

		#----------------------------------------------
		#	===== Plot Results =====
		f, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
		ax1.stem(w_marker, marker_f_tremor)
		ax1.set_title('Background Marker Tremor')
		ax2.stem(w_stab, stab_f_tremor)
		ax2.set_title('Stabilized Marker Tremor')
		plt.show()


		fig = plt.figure()
		ax = Axes3D(fig)
		ax.plot(marker_x, marker_y, c="b",  zs=marker_z)
		ax.plot(stab_x, stab_y,  c="r", zs=stab_z)
		plt.show()

	toolFlag = True
	if(toolFlag):
		#--------------------------------------------
		# 	====== Marker Frequency Analysis =====
		# link nearby points into trajectories
		toolTraj = buildTrajectory(toolPoints)
		# remove trajectories < 4 pts long
		tool_X = pruneTraj(toolTraj)
		# get one tool trajectory 
		tool_X_linked = linkToolTrajs(tool_X)
		# get x, y, z trajectories
		tool_x, tool_y, tool_z = buildXYZ_Trajectories([tool_X_linked])
		# frequency analysis
		tool_f_tremor, w_tool = frequencyAnalysis([tool_x, tool_y, tool_z], Fs/2)


		#--------------------------------------------
		# 	====== Stabilized Frequency Analysis =====
		# link nearby points into trajectories
		stabTraj = buildTrajectory(tool_stable)
		# remove trajectories < 4 pts long
		stab_X = pruneTraj(stabTraj)
		# get one tool trajectory 
		stab_X_linked = linkToolTrajs(stab_X)
		# get x, y, z trajectories
		stab_x, stab_y, stab_z = buildXYZ_Trajectories([stab_X_linked])
		# frequency analysis
		stab_f_tremor, w_stab = frequencyAnalysis([stab_x, stab_y, stab_z], Fs/2)


		pdb.set_trace()

		#----------------------------------------------
		#	===== Plot Results =====
		f, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
		ax1.stem(w_tool, tool_f_tremor)
		ax1.set_title('Tool Tremor')
		ax2.stem(w_stab, stab_f_tremor)
		ax2.set_title('Stabilized Tool Tremor')
		plt.show()


		fig = plt.figure()
		ax = Axes3D(fig)
		ax.plot(tool_x, tool_y, c="b",  zs=tool_z)
		ax.plot(stab_x, stab_y,  c="r", zs=stab_z)
		plt.show()
