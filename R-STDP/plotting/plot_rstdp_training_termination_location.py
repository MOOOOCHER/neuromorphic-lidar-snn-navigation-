import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec

path = "../data/session_xyz"
h5f = h5py.File(path + '/' + 'rstdp_data.h5', 'r')
w_l = np.array(h5f['w_l'], dtype=float)
w_r = np.array(h5f['w_r'], dtype=float)
w_i = np.array(h5f['w_i'], dtype=float)
e_i = np.array(h5f['e_i'], dtype=float)

veh_loc_innerLane = np.array(h5f['veh_loc_i'], dtype=float)
veh_loc_outerLane = np.array(h5f['veh_loc_o'], dtype=float)
veh_loc_innerLane = np.round(veh_loc_innerLane, decimals=2)
veh_loc_outerLane = np.round(veh_loc_outerLane, decimals=2)
# Create a figure and axis object
columns = ('x-Position', 'y-Position', 'z-Position')

# Create a figure and axes object
fig = plt.figure(figsize=(6, 13))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])

ax1 = plt.subplot(gs[0])
# Create the table
table = ax1.table(cellText=veh_loc_innerLane, colLabels=columns, loc='center')
# Format the table
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1, 1.5)
# Hide the axes
ax1.axis('off')

ax2 = plt.subplot(gs[1])
# Create the table
table = ax2.table(cellText=veh_loc_outerLane, colLabels=columns, loc='center')
# Format the table
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1, 1.5)
# Hide the axes
ax2.axis('off')

ax1.set_title('Inner Lane Termination Position', y=1.05)
ax2.set_title('Outer Lane Termination Position', y=1.05)

plt.savefig(path + '/' + 'rstdp_data_termination' + '.pdf', dpi=300)
# Display the table
plt.show()
