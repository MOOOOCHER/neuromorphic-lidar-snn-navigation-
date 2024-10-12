#!/usr/bin/env python

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import sem

# R-STDP training progress
# Fig. 5.6, Fig. 5.9

path = "../data/session_xyz"
h5f = h5py.File(path + '/rstdp_data.h5.clockwise.8x4.2_performance_data.2.h5', 'r')
d = np.array(h5f['distance'], dtype=float)
h5f = h5py.File(path + '/rstdp_data.h5.clockwise.4x4.3_performance_data.2.h5', 'r')
d2 = np.array(h5f['distance'], dtype=float)
# d = d[0:2500]

plt.style.use('seaborn')
plt.rcParams.update({'legend.fontsize': 14})
plt.rcParams.update({'xtick.labelsize': 16})
plt.rcParams.update({'ytick.labelsize': 16})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'axes.titlesize': 16})
plt.rcParams.update({'axes.labelsize': 17})
plt.rcParams.update({'lines.linewidth': 3})
plt.rcParams.update({'lines.markersize': 6})
plt.rcParams.update({'mathtext.fontset': 'stix'})
plt.rcParams.update({'font.family': 'STIXGeneral'})
plt.rcParams.update({'font.size': 15})

plt.rcParams['pdf.fonttype'] = 42

def prep(d):
    start_dis = 0
    for i in range(len(d)):
        step = d[i][4]
        distance = d[i][3]
        if step == 1:
            start_dis = distance
            print(distance)
        d[i][3] -= start_dis

    d = d[d[:, 3].argsort()]
    return d

d = prep(d)
d2 = prep(d2)

def calc(d):
    band = []
    slice_width = 1
    for i in np.arange(0, 200, slice_width):
        delta_d = []
        for j in d:
            if j[3] > i + slice_width:
                break
            if j[3] > i - slice_width:
                delta_d.append(j[2])
        if not delta_d:
            continue
        band.append([i, np.mean(delta_d), np.max(delta_d), np.min(delta_d), np.std(delta_d)])
    band = np.array(band)
    return band

band = calc(d)
band2 = calc(d2)

# band2 = []
# for i in np.arange(0, 200, 1):
#     delta_d = []
#     for j in band:
#         if j[0] > i + 1:
#             break
#         if j[0] > i - 1:
#             delta_d.append(j[1])
#     if not delta_d:
#         continue
#     band2.append([i, np.mean(delta_d), np.max(delta_d), np.min(delta_d)])
# band2 = np.array(band2)

fig = plt.figure(figsize=(7.2, 3))
gs = gridspec.GridSpec(1, 1, height_ratios=[2])

ax1 = plt.subplot(gs[0])
ax1.set_ylabel('Lane Deviation [m]')
ax1.set_xlabel('Traveled Distance [m]')
ax1.set_xlim((0, 190))
ax1.set_ylim((-1, 1))
ax1.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True)
# plt.scatter(d[:, 3], d[:, 2])
# plt.scatter(band[:, 0], band[:, 1])
# plt.scatter(band[:, 0], band[:, 2])
# plt.scatter(band[:, 0], band[:, 3])

plt.plot(band[:, 0], band[:, 1], label="8x4 Sensory Neurons")
plt.fill_between(band[:, 0], band[:, 1] - band[:, 4], band[:, 1] + band[:, 4], alpha=0.4)

# plt.fill_between(band[:, 0], band[:, 3], band[:, 2], alpha=0.3)
plt.plot(band2[:, 0], band2[:, 1], color='C2', label="4x4 Sensory Neurons")
plt.fill_between(band2[:, 0], band2[:, 1] - band2[:, 4], band2[:, 1] + band2[:, 4], alpha=0.4, color='C2')

ax1.legend()

# plt.plot(band2[:, 0], band2[:, 1])
# plt.fill_between(band2[:, 0], band2[:, 3], band2[:, 2], alpha=0.2)

print('mean: {}, rmse: {}'.format(np.mean(np.absolute(d[:, 2])), np.sqrt(np.mean((d[:, 2])**2))))

fig.tight_layout()
plt.subplots_adjust(wspace=0., hspace=0.1)
plt.savefig(path + '/error.bands.pdf', dpi=300)
plt.show()
