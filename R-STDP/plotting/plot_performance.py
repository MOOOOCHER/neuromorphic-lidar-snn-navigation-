#!/usr/bin/env python
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec

# R-STDP training progress
# Fig. 5.6, Fig. 5.9

path = "../data/session_xyz"
h5f = h5py.File(path + '/Ground Truth Data/' + 'scenario2_performance_data_nengo.h5', 'r')
d = np.array(h5f['distance'], dtype=float)
episode1 = d[d[:,1]==0]
episode2 = d[d[:,1]==1]
episode3 = d[d[:,1]==2]
episode4 = d[d[:,1]==3]

veh_loc = np.array(h5f['position_termination'], dtype=float)
avg_velocity = np.array(h5f['velocity'], dtype=float)
sim_time = np.array(h5f['sim_time'], dtype=float)
sim_time_ep1 = sim_time[int(episode1[0,0]):int(episode1[-1,0])]
sim_time_ep2 = sim_time[int(episode2[0,0]):int(episode2[-1,0])]
sim_time_ep3 = sim_time[int(episode3[0,0]):int(episode3[-1,0])]
sim_time_ep4 = sim_time[int(episode4[0,0]):int(episode4[-1,0])]

plt.style.use('seaborn')
plt.rcParams.update({'xtick.labelsize': 16})
plt.rcParams.update({'ytick.labelsize': 16})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'axes.titlesize': 16})
plt.rcParams.update({'axes.labelsize': 17})
plt.rcParams.update({'lines.linewidth': 2})
plt.rcParams.update({'lines.markersize': 6})
plt.rcParams.update({'mathtext.fontset': 'stix'})
plt.rcParams.update({'font.family': 'STIXGeneral'})
plt.rcParams.update({'font.size': 15})

plt.rcParams['pdf.fonttype'] = 42

fig = plt.figure(figsize=(9, 13))
gs = gridspec.GridSpec(4, 1, height_ratios=[3,3,3,3])

ax1 = plt.subplot(gs[0])
ax1.set_ylabel('Deviation [m]')
ax1.set_xlabel('Simulation Step\n'+ 'Mean: {}, RMSE: {}'.format(np.mean(np.absolute(episode1[:, 2])).round(3), np.sqrt(np.mean((episode1[:, 2])**2)).round(3))
               +', Avg Velocity: '+ str(round(avg_velocity[0],3))
               +'\nAvg Simulation Time per Step: '+ str(round(np.mean(sim_time_ep1)*1000,3))+"ms")
ax1.set_xlim((episode1[0,0],episode1[-1,0]))
ax1.set_ylim((-2, 2))
ax1.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True)
plt.plot(episode1[:, 0], episode1[:, 2])

ax2 = plt.subplot(gs[1])
ax2.set_ylabel('Deviation [m]')
ax2.set_xlabel('Simulation Step\n'+ 'Mean: {}, RMSE: {}'.format(np.mean(np.absolute(episode2[:, 2])).round(3), np.sqrt(np.mean((episode2[:, 2])**2)).round(3))
               +', Avg Velocity: '+ str(round(avg_velocity[1],3))
               + '\nAvg Simulation Time per Step: ' + str(round(np.mean(sim_time_ep2) * 1000, 3))+"ms")
ax2.set_xlim((0,episode2[-1,0]-episode2[0,0]))
ax2.set_ylim((-2, 2))
ax2.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True)
plt.plot(episode2[:, 0]-episode2[0,0], episode2[:, 2])

ax3 = plt.subplot(gs[2])
ax3.set_ylabel('Deviation [m]')
ax3.set_xlabel('Simulation Step\n'+ 'Mean: {}, RMSE: {}'.format(np.mean(np.absolute(episode3[:, 2])).round(3), np.sqrt(np.mean((episode3[:, 2])**2)).round(3))
               +', Avg Velocity: '+ str(round(avg_velocity[2],3))
               + '\nAvg Simulation Time per Step: ' + str(round(np.mean(sim_time_ep3) * 1000, 3))+"ms")
ax3.set_xlim((0,episode3[-1,0]-episode3[0,0]))
ax3.set_ylim((-2, 2))
ax3.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True)
plt.plot(episode3[:, 0]-episode3[0,0], episode3[:, 2])


ax4 = plt.subplot(gs[3])
ax4.set_ylabel('Deviation [m]')
ax4.set_xlabel('Simulation Step\n'+ 'Mean: {}, RMSE: {}'.format(np.mean(np.absolute(episode4[:, 2])).round(3), np.sqrt(np.mean((episode4[:, 2])**2)).round(3))
                +', Avg Velocity: '+ str(round(avg_velocity[3],3))
               + '\nAvg Simulation Time per Step: ' + str(round(np.mean(sim_time_ep4) * 1000, 3)) +"ms")
ax4.set_xlim((0,episode4[-1,0]-episode4[0,0]))
ax4.set_ylim((-2, 2))
ax4.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True)
plt.plot(episode4[:, 0]-episode4[0,0], episode4[:, 2])


fig.tight_layout()
plt.subplots_adjust(wspace=0., hspace=0.1)
plt.savefig(path + '/Ground Truth Data/' + 'scenario2_performance_data_nengo' + '.pdf', dpi=300)
plt.show()
