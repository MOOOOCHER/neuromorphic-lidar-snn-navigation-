#!/usr/bin/env python

import numpy as np

path = './data/session_xyz'			# Path for saving data

# Input image
dvs_resolution = [128,128]			# Original DVS frame resolution
crop_top = 40						# Crop at the top
crop_bottom = 24					# Crop at the bottom
resolution = [8, 4]					# Resolution of reduced image
                                    # newstate = [4,4]; newstate_new = [4,8]; fcn = [8,8]; fcn_right = [4,4]

# Network parameters
sim_time = 50.0						# Length of network simulation during each step in ms
t_refrac = 2.						# Refractory period
time_resolution = 0.1				# Network simulation time resolution
iaf_params = {}						# IAF neuron parameters
poisson_params = {}					# Poisson neuron parameters
max_poisson_freq = 300.				# Maximum Poisson firing frequency for n_max  # 300.  #FCN_8shape: 225.
max_spikes = 4.3	 				# number of events during each step for maximum poisson frequency
                                    # newstate = 130.; # newstate_new = [20., 22.]; # newstate_fcn = 5.3; # newstate_diff (X:160,200,250,130,90); newstate_fcn_right = 8.9
# R-STDP parameters
w_min = 0.							# Minimum weight value
w_max = 3000.						# Maximum weight value
w0_min = 200.						# Minimum initial random value
w0_max = 201.						# Maximum initial random value
tau_n = 200.						# Time constant of reward signal
tau_c = 1000.						# Time constant of eligibility trace
reward_factor = 0.002				# Reward factor modulating reward signal strength  0.001
A_plus = 1.0							# Constant scaling strength of potentiation   0.2
A_minus = 1.0					# Constant scaling strength of depression   0.2

# Steering wheel model
v_max = 0.4						# Maximum speed   # 0.4
v_min = 0.2							# Minimum speed   # 0.2
turn_factor = 1.2					# Factor controls turn radius
turn_pre = 0						# Initial turn speed
v_pre = v_max						# Initial speed
n_max = sim_time//t_refrac - 10.	# Maximum input activity (50//2-10 = 15)

# Other
reset_distance = 2.0				# Reset distance   # 0.25   FCN_right:0.15
rate = 20.							# ROS publication rate motor speed
training_length = 20000			# Length of training procedure (1 step ~ 50 ms)   #100000  #fcn:18000
max_step = 10000                 # For Training:600, For scenario 1: 10000, For scenario 2: 10000
