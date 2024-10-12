import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec

path = "../data/session_xyz"
h5f = h5py.File(path + '/' + 'FCN Data/scenario1_performance_data_nest.h5', 'r')

veh_loc = np.array(h5f['position_termination'], dtype=float)
veh_loc = veh_loc.round(2)
# Create a figure and axis object
columns = ('x-Position', 'y-Position', 'z-Position')

# Create a figure and axes object
fig = plt.figure(figsize=(5, 2.5))
gs = gridspec.GridSpec(1, 1, height_ratios=[3])

ax1 = plt.subplot(gs[0])
# Create the table
table = ax1.table(cellText=veh_loc, colLabels=columns, loc='center')
# Format the table
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1, 1.5)
# Hide the axes
ax1.axis('off')
ax1.set_title('Termination Position', y=0.9)

plt.savefig(path + '/' + 'FCN Data/performance_termination_nest' + '.pdf', dpi=300)
# Display the table
plt.show()
