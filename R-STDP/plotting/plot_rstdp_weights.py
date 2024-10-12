#!/usr/bin/env python

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

# R-STDP weights learned
# Fig. 5.7, Fig. 5.8, Fig. 5.10

path = "../data/session_xyz"
h5f = h5py.File(path + '/' + 'rstdp_data.h5', 'r')

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

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"]

plt.rcParams['pdf.fonttype'] = 42

w_l = np.array(h5f['w_l'], dtype=float)
w_r = np.array(h5f['w_r'], dtype=float)
print(w_l.shape)
weights_l = np.flipud(w_l[-1].T)
weights_r = np.flipud(w_r[-1].T)
max_weight = max(weights_l.max(), weights_r.max())
print("max", max_weight)


def boost(w):
	base = 10
	max = np.max(w)
	w = w / np.max(w) * (base - 1)
	w = 30 * np.log(1 + w) / np.log(base)
	w = w / np.max(w) * max
	return w

fig = plt.figure(figsize=(7.2, 4))

ax1 = plt.subplot(121)
plt.title('Left Weights')
plt.imshow(boost(weights_l), alpha=0.7, cmap='PuBu', vmax=max_weight, aspect='auto')
plt.axis('off')
for (j,i),label in np.ndenumerate(weights_l):
	ax1.text(i,j,int(label),ha='center',va='center')

ax2 = plt.subplot(122)
plt.title('Right Weights')
plt.imshow(boost(weights_r), alpha=0.7, cmap='PuBu', vmax=max_weight, aspect='auto')
plt.axis('off')
for (j,i),label in np.ndenumerate(weights_r):
	ax2.text(i,j,int(label),ha='center',va='center')

fig.tight_layout()
plt.savefig(path + '/' + 'rstdp_data.h5.weights.pdf', dpi=300)
plt.show()
