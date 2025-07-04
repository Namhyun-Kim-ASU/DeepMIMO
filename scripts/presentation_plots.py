
#%% 

import deepmimo as dm

dataset = dm.load('asu_campus_3p5')
import matplotlib.pyplot as plt

params = dm.ChannelParameters()
params.bs_antenna.rotation = [0, 0, -135]
dataset.set_channel_params(params)

#%% Plot converage

plt.rcdefaults()
plt.rcParams['text.color'] = 'white'           # Font color
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['figure.facecolor'] = 'grey'     # Background outside plot
plt.rcParams['axes.facecolor'] = 'white'       # Background inside plot

attrs = ['power', 'phase', 'delay', 'aoa_az', 'aoa_el', 'aod_az', 'aod_el' ]

for attr in attrs:
    dataset[attr].plot(dpi=300)
    plt.savefig(f"dm_coverage_{attr}.png", transparent=True)

#%% Plot LoS

dataset.los = dataset.los.astype(float)
dataset.los[dataset.los == -1] = np.nan


dataset.los.plot(dpi=300)
plt.savefig("dm_los.png", transparent=True)

#%% Plot rays

plt.rcdefaults()
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
dataset.plot_rays(55039, dpi=300)
plt.savefig("plot_rays.png", transparent=True)

#%% Plot scene
plt.rcParams['text.color'] = 'white'
plt.rcParams['font.size'] = 16
dataset.scene.plot(dpi=300)
# (additional changing terrain color to black in scene.py)

plt.savefig("dm_scene.png", transparent=True)

#%% Doppler Sequence (Static)

import numpy as np

plt.rcdefaults()
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['font.size'] = 16

seq_idxs1 = dm.LinearPath(dataset.rx_pos, [150, -120], [150, 80], n_steps=200).idxs
seq_idxs2 = dm.LinearPath(dataset.rx_pos, [150, 85], [-120, 85], n_steps=200).idxs
seq_idxs = np.concatenate([seq_idxs1, seq_idxs2])

n = 8
ax = dataset.scene.plot(dpi=300, proj_3D=False, title='')
ax.scatter(dataset.rx_pos[seq_idxs[::n], 0], dataset.rx_pos[seq_idxs[::n], 1],
           s=40, facecolors=(1, 0, 0, 0.5), edgecolors='black', 
           label='User Positions')
plt.legend(loc='upper right')
ax.set_facecolor((1, 1, 1, 0.5))
ax.figure.patch.set_alpha(0.0)  # fully transparent outer background
plt.savefig("dm_scene_doppler_static.png")

#%% Doppler Sequence (Dynamic)

proj_3D = True
ax = dataset.scene.plot(dpi=300, proj_3D=proj_3D, title='')
# plt.xlim((0, 200))
# plt.ylim((0, 200))
dataset.plot_rays(16816, ax=ax, proj_3D=proj_3D)
# Option 1: Directly Edit objects in the scene
#    - delete unnecessary buildings
#    - reduce the size of the size of the floor
# Option 2: Add arguments in the scene.plot() to only include objects within the ranges

# plt.savefig("dm_scene.png", transparent=True)
