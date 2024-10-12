#!/usr/bin/env python

# import numpy as np
import sys
from environment import *
from nest_network import *
from parameters import *
import h5py
import time

generateLidarLabels = False
useFCN = True
scenario="scenario1"
env = CarlaEnvironment(generateLidarLabels, useFCN, isTraining=False, scenario=scenario)
time_sim_inference = []
time_step = []
# Read network weights
h5f = h5py.File(path + '/rstdp_data.h5' , 'r') # for 8-loop
w_l = np.array(h5f['w_l'], dtype=float)[-1]
w_r = np.array(h5f['w_r'], dtype=float)[-1]
snn = SpikingNeuralNetwork()
# Set network weights
snn.set_weights(w_l,w_r)
veh_loc_termination = []
distance = []
velocities = []
episode_velocity = []

episode = 0
m = 0
# Initialize environment, get state, get reward
s,r = env.reset()

for i in range(p.training_length):

    # Simulate network for 50 ms
    # Get left and right output spikes, get weights
    sim_start = time.time()
    n_l, n_r, w1, w2 = snn.simulate(s, r)
    sim_end = time.time()
    time_sim_inference.append(sim_end - sim_start)
    # Feed output spikes into steering wheel model
    # Get state, distance, position, reward, termination, step, lane
    step_start = time.time()
    s, d, p, r, t, n,steps,if_reset, veh_loc, velocity_step = env.step(n_l, n_r, episode)
    step_end = time.time()
    time_step.append(step_end-step_start)
    episode_velocity.append(velocity_step)
    # Break episode if robot reaches starting position again
    # if p == env.d_outer or p == env.d_inner:
    # Store position, distance
    distance.append([i, episode, d, p, n])
    if t:
        episode = episode + 1
        veh_loc_termination.append([veh_loc.x, veh_loc.y, veh_loc.z])
        velocities.append(np.mean(np.array(episode_velocity)))
        episode_velocity = []
        if episode == 4:
            break

    if i % 50 == 0:
        # Save performance data
        if scenario=="scenario2":
            h5f = h5py.File(path + '/scenario2' + '_performance_data_nest.h5', 'w')
        else:
            h5f = h5py.File(path + '/scenario1' + '_performance_data_nest.h5', 'w')
        h5f.create_dataset('distance', data=distance)
        h5f.create_dataset('position_termination', data=veh_loc_termination)
        h5f.create_dataset('velocity', data=velocities)
        h5f.create_dataset('sim_time', data=time_sim_inference)
        h5f.close()


if scenario == "scenario2":
    h5f = h5py.File(path + '/scenario2' + '_performance_data_nest.h5', 'w')
else:
    h5f = h5py.File(path + '/scenario1' + '_performance_data_nest.h5', 'w')
h5f.create_dataset('distance', data=distance)
h5f.create_dataset('position_termination', data=veh_loc_termination)
h5f.create_dataset('velocity', data=velocities)
h5f.create_dataset('sim_time', data=time_sim_inference)
h5f.close()

print('Avg Sim-Time: '+str(np.mean(np.array(time_sim_inference)))+ ", Avg Step-Time: "+ str(np.mean(np.array(time_step))))
