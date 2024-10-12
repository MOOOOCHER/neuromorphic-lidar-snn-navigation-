#!/usr/bin/env python

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec

# R-STDP training progress
# Fig. 5.6, Fig. 5.9

path = "../data/session_xyz"
h5f = h5py.File(path + '/' + 'rstdp_data.h5', 'r')
w_l = np.array(h5f['w_l'], dtype=float)
w_r = np.array(h5f['w_r'], dtype=float)
w_i = np.array(h5f['w_i'], dtype=float)
e_i = np.array(h5f['e_i'], dtype=float)
e_p_innerLane = np.array(h5f['e_p_i'], dtype=float)
e_p_outerLane = np.array(h5f['e_p_o'], dtype=float)
s_c_innerLane = np.array(h5f['s_c_i'], dtype=float)
s_c_outerLane = np.array(h5f['s_c_o'], dtype=float)
veh_loc_innerLane = np.array(h5f['veh_loc_i'], dtype=float)
veh_loc_outerLane = np.array(h5f['veh_loc_o'], dtype=float)

# start_i = 0
# for i in range(len(s_c)):
# 	if s_c[i][0] > start_i:
# 		start_i = s_c[i][0]
# 	elif s_c[i][0] < start_i:
# 		s_c[i][0] += start_i
# 	if s_c[i][1] > 550:
# 		s_c[i][1] = 550
# start_i = 0
# for i in range(len(w_i)):
# 	if w_i[i] > start_i:
# 		start_i = w_i[i]
# 	elif w_i[i] < start_i:
# 		w_i[i] += start_i
#
# h5f = h5py.File(path + '/rstdp_data.h5', 'w')
# h5f.create_dataset('w_l', data=w_l)
# h5f.create_dataset('w_r', data=w_r)
# h5f.create_dataset('w_i', data=w_i)
# h5f.create_dataset('e_p', data=e_p)
# h5f.create_dataset('e_i', data=e_i)
# h5f.create_dataset('s_c', data=s_c)
# h5f.close()


xlim = 20000
ylim = 4000
# ylim = 1000
# xlim = w_i[-1]
print("i ", xlim)

plt.style.use('seaborn')
# xtick.labelsize: 16
# ytick.labelsize: 16
# font.size: 15
# figure.autolayout: True
# figure.figsize: 7.2,4.45
# axes.titlesize : 16
# axes.labelsize : 17
# lines.linewidth : 2
# lines.markersize : 6
# legend.fontsize: 13
# mathtext.fontset: stix
# font.family: STIXGeneral
plt.rcParams.update({'xtick.labelsize': 16})
plt.rcParams.update({'ytick.labelsize': 16})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'axes.titlesize': 16})
plt.rcParams.update({'axes.labelsize': 17})
plt.rcParams.update({'lines.linewidth': 2})
plt.rcParams.update({'lines.markersize': 6})
# plt.rcParams.update({'mathtext.fontset': 'stix'})
# plt.rcParams.update({'font.family': 'STIXGeneral'})
plt.rcParams.update({'font.size': 15})

plt.rcParams['pdf.fonttype'] = 42

fig = plt.figure(figsize=(9, 13))
gs = gridspec.GridSpec(4, 1, height_ratios=[3, 3, 3, 3])

ax1 = plt.subplot(gs[0])
ax1.set_ylabel('Steps per episode.\nInner Loop')
ax1.set_xlim((0,xlim))
ax1.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True)
plt.scatter(s_c_innerLane[:,0], s_c_innerLane[:,1])

ax2 = plt.subplot(gs[1])
ax2.set_ylabel('Steps per episode\nOuter Loop')
ax2.set_xlim((0,xlim))
ax2.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True)
plt.scatter(s_c_outerLane[:,0], s_c_outerLane[:,1])

ax3 = plt.subplot(gs[2])
ax3.set_ylabel('Distance [m]\nInner Loop')
ax3.set_xlim((0,xlim))
ax3.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True)
plt.scatter(e_p_innerLane[:,0], e_p_innerLane[:,1])

ax4 = plt.subplot(gs[3],sharey=ax3)
ax4.set_ylabel('Distance [m]\nOuter Loop')
ax4.set_xlim((0,xlim))
ax4.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True)
plt.scatter(e_p_outerLane[:,0], e_p_outerLane[:,1])

ax4.set_xlabel('Simulation Time [1 step = 50 ms].')


fig.tight_layout()
plt.subplots_adjust(wspace=0., hspace=0.1)
plt.savefig(path + '/' + 'rstdp_data_distance' + '.pdf', dpi=300)
plt.show()
