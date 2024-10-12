#!/usr/bin/env python

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt


# plt.style.use('seaborn')
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

features = np.array([
	[0.2, 1., 0.52, 0.3],
	[0.8, 0.1, 1.1, 0.27],
	[1.3, 0.13, 1., 0.21]
])

plt.imshow(features, alpha=0.9, cmap='Blues')
plt.axis('off')

plt.savefig('./lidar_lane_feature.svg', dpi=300)
plt.show()