#!/usr/bin/env python

# import numpy as np
from environment import *
from nest_network import *
from parameters import *
import h5py
import argparse

snn = SpikingNeuralNetwork()
env = CarlaEnvironment(False, False, True)
weights_r = []
weights_l = []
weights_i = []
episode_position_i = []
episode_position_o = []
episode_i = []
step_count_inner = []
step_count_outer = []
veh_loc_i = []
veh_loc_o = []
episode = 0
# Initialize environment, get initial state, initial reward
s, r = env.reset()

start_i = 0

def save_data():
	# Save data
	h5f = h5py.File(path + '/rstdp_data.h5', 'w')
	h5f.create_dataset('w_l', data=weights_l)
	h5f.create_dataset('w_r', data=weights_r)
	h5f.create_dataset('w_i', data=weights_i)
	h5f.create_dataset('e_p_i', data=episode_position_i)
	h5f.create_dataset('e_p_o', data=episode_position_o)
	h5f.create_dataset('e_i', data=episode_i)
	h5f.create_dataset('s_c_i', data=step_count_inner)
	h5f.create_dataset('s_c_o', data=step_count_outer)
	h5f.create_dataset('veh_loc_i', data=veh_loc_i)
	h5f.create_dataset('veh_loc_o', data=veh_loc_o)

	h5f.close()


def load_data():
	# Read network weights
	global w_l, w_r
	h5f = h5py.File(path + '/rstdp_data.h5', 'r')

	global weights_l, weights_r, weights_i, episode_position_i, episode_position_o, episode_i, step_count_inner, step_count_outer, veh_loc_i, veh_loc_o, start_i
	weights_l = np.array(h5f['w_l'], dtype=float).tolist()
	weights_r = np.array(h5f['w_r'], dtype=float).tolist()
	weights_i = np.array(h5f['w_i'], dtype=float).tolist()
	episode_position_i = np.array(h5f['e_p_i'], dtype=float).tolist()
	episode_position_o = np.array(h5f['e_p_o'], dtype=float).tolist()
	episode_i = np.array(h5f['e_i'], dtype=float).tolist()
	step_count_inner = np.array(h5f['s_c_i'], dtype=float).tolist()
	step_count_outer = np.array(h5f['s_c_o'], dtype=float).tolist()
	veh_loc_i = np.array(h5f['veh_loc_i'], dtype=float).tolist()
	veh_loc_o = np.array(h5f['veh_loc_o'], dtype=float).tolist()
	start_i = weights_i[-1]

	w_l = np.array(h5f['w_l'], dtype=float)[-1]
	w_r = np.array(h5f['w_r'], dtype=float)[-1]
	# Set network weights
	snn.set_weights(w_l,w_r)
	h5f.close()


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-c', action='store_true')
args = parser.parse_args()

if args.c:
	load_data()
	print("Continue Training")
for i in range(training_length):

	# Simulate network for 50 ms
	# get number of output spikes and network weights
	n_l, n_r, w_l, w_r = snn.simulate(s, r)

	# Feed output spikes in steering wheel model
	# get state, distance, position, reward, termination, steps, reset
	s, d, p, r, t, n, steps, if_reset, veh_loc, _ = env.step(n_l, n_r, episode)

	if if_reset:
		if env.start_innerLane:
			step_count_inner.append([start_i + i, steps])
		else:
			step_count_outer.append([start_i + i, steps])
		print("Reset, i ", start_i + i)

	# Save weights every 100 simulation steps
	if i % 100 == 0:
		weights_l.append(w_l)
		weights_r.append(w_r)
		weights_i.append(start_i + i)
		save_data()

	# Save last position if episode is terminated
	if t:
		if env.start_innerLane:
			episode_position_i.append([start_i + i,p])
			veh_loc_i.append([veh_loc.x, veh_loc.y, veh_loc.z])
		else:
			episode_position_o.append([start_i + i,p])
			veh_loc_o.append([veh_loc.x, veh_loc.y, veh_loc.z])

		episode_i.append(start_i + i)
		episode = episode + 1
		print("Episode: " + str(episode) + "    Total steps: " + str(i) + "    Steps: " + str(n) + '	Distance traveled this episode: ' + str(p))
