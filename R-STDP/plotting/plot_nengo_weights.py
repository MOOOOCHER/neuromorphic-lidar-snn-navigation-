
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
sys.path.append('..')
from nengo_network import NengoLoihiSpikingNeuralNetwork
# R-STDP weights learned
# Fig. 5.7, Fig. 5.8, Fig. 5.10

path = "../data/session_xyz"
h5f = h5py.File(path + '/' + 'rstdp_data.h5', 'r')
w_l = np.array(h5f['w_l'], dtype=float)[-1]
w_r = np.array(h5f['w_r'], dtype=float)[-1]
nengo_network = NengoLoihiSpikingNeuralNetwork(w_l,w_r)
# plt.style.use('seaborn')
nengo_network.sim.run(0.05)
weightsL = []
weightsR = []
for i in range(32):
    weightsL.append(nengo_network.sim.data[nengo_network.connections_l[i]].weights.toarray()[0][0])
    weightsR.append(nengo_network.sim.data[nengo_network.connections_r[i]].weights.toarray()[0][0])
weightsL = np.array(weightsL).reshape(8, 4)
weightsR = np.array(weightsR).reshape(8, 4)


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

weights_l = np.flipud(weightsL.T)
weights_r = np.flipud(weightsR.T)
weights_l = np.round(weights_l,3)
weights_r = np.round(weights_r,3)

fig = plt.figure(figsize=(10, 4))

ax1 = plt.subplot(121)
plt.title('Left Weights')
plt.imshow(weights_l, alpha=0.7, cmap='PuBu', aspect='auto')
plt.axis('off')
for (j,i),label in np.ndenumerate(weights_l):
	ax1.text(i,j,label,ha='center',va='center')

ax2 = plt.subplot(122)
plt.title('Right Weights')
plt.imshow(weights_r, alpha=0.7, cmap='PuBu', aspect='auto')
plt.axis('off')
for (j,i),label in np.ndenumerate(weights_r):
	ax2.text(i,j,label,ha='center',va='center')

fig.tight_layout()
plt.savefig(path + '/' + 'nengo_weights.pdf', dpi=300)
plt.show()
