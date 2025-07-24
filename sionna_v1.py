
#%% Imports


import deepmimo as dm
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, DirectivePattern, PathSolver

def compute_array_combinations(arrays):
    return np.stack(np.meshgrid(*arrays), -1).reshape(-1, len(arrays))

def gen_user_grid(box_corners, steps, box_offsets=None):
    """
    box_corners is = [bbox_min_corner, bbox_max_corner]
    steps = [x_step, y_step, z_step]
    """

    # Sample the ranges of coordinates
    ndim = len(box_corners[0])
    dim_ranges = []
    for dim in range(ndim):
        if steps[dim]:
            dim_range = np.arange(box_corners[0][dim], box_corners[1][dim], steps[dim])
        else:
            dim_range = np.array([box_corners[0][dim]]) # select just the first limit

        dim_ranges.append(dim_range + box_offsets[dim] if box_offsets else 0)

    pos = compute_array_combinations(dim_ranges)
    print(f'Total positions generated: {pos.shape[0]}')
    return pos


# Create antenna array
tx_array = PlanarArray(num_rows=2,
                           num_cols=2,
                           vertical_spacing=0.5,
                           horizontal_spacing=0.5,
                           pattern="iso",
                           polarization="V")

rx_array = PlanarArray(num_rows=1,
                           num_cols=1,
                           vertical_spacing=0.5,
                           horizontal_spacing=0.5,
                           pattern="iso",
                           polarization="V")

def create_base_scene(scene_path, center_frequency):
    scene = load_scene(scene_path)
    scene.frequency = center_frequency
    scene.tx_array = tx_array
    scene.rx_array = rx_array
    scene.synthetic_array = True
    return scene

from deepmimo.exporters.sionna_exporter import export_paths

#%% Advanced Example (NOT WORKING WITH GPU ON UBUNTU 24.04)
# Path solver setup
path_solver = PathSolver()

# Save dict with compute path params to export later  
my_compute_path_params = dict(
    max_depth=5,
    los=True,
    specular_reflection=True,
    diffuse_reflection=False,
    refraction=True,
    synthetic_array=True,
    seed=42
)
carrier_freq = 3.5 * 1e9  # Hz

tx_pos = [-33, 11, 32.03]

# 0- Create/Fetch scene and get buldings in the scene
scene = create_base_scene(sionna.rt.scene.simple_street_canyon,
                          center_frequency=carrier_freq)

# 1- Compute TX position
print('Computing BS position')
tx = Transmitter(name="tx", position=tx_pos, orientation=[0,0,0])
tx.antenna = tx_array
scene.add(tx)

# 2- Compute RXs positions
print('Computing UEs positions')
rxs = gen_user_grid(box_corners=[(-93, -60, 0), (93, 60, 0)],
                    steps=[4, 4, 0], box_offsets=[0, 0, 2])

# 3- Add the first batch of receivers to the scene
n_rx = len(rxs)
n_rx_in_scene = 10  # to compute in parallel
print(f'Adding users to the scene ({n_rx_in_scene} at a time)')
for rx_idx in range(n_rx_in_scene):
    rx = Receiver(name=f"rx_{rx_idx}", position=rxs[rx_idx], orientation=[0,0,0])
    rx.antenna = rx_array
    scene.add(rx)

# 4- Enable scattering in the radio materials
if my_compute_path_params.get('diffuse_reflection', False):
    for rm in scene.radio_materials.values():
        rm.scattering_coefficient = 1/np.sqrt(3) # [0,1]
        rm.scattering_pattern = DirectivePattern(alpha_r=10)

# 5- Compute the paths for each set of receiver positions
path_list = []
n_rx_remaining = n_rx
for x in tqdm(range(int(n_rx / n_rx_in_scene)+1), desc='Path computation'):
    if n_rx_remaining > 0:
        n_rx_remaining -= n_rx_in_scene
    else:
        break
    if x != 0:
        # modify current RXs in scene
        for rx_idx in range(n_rx_in_scene):
            if rx_idx + n_rx_in_scene*x < n_rx:
                scene.receivers[f'rx_{rx_idx}'].position = rxs[rx_idx + n_rx_in_scene*x]
            else:
                # remove the last receivers in the scene
                scene.remove(f'rx_{rx_idx}')

    paths = path_solver(scene=scene, **my_compute_path_params)
    paths.normalize_delays = False

    path_list.append(export_paths(paths)[0])

#%% Simple Example (WORKING WITH GPU ON UBUNTU 24.04)
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver
# scene = load_scene(sionna.rt.scene.simple_street_canyon, merge_shapes=False)
scene = load_scene(merge_shapes=False)
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=3,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="V")

scene.rx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="V")

tx = Transmitter(name="tx", position=[5, 0, 0])
scene.add(tx)

rx = Receiver(name="rx", position=[30, 0, 0])
rx2 = Receiver(name="rx2", position=[40, 0, 5])
scene.add(rx)
scene.add(rx2)

p_solver  = PathSolver()
my_compute_path_params = dict(
    max_depth=0,
    # max_depth=3,
    synthetic_array=False
)
paths = p_solver(scene=scene, **my_compute_path_params)
path_list = [export_paths(paths)[0]]

#%%

# Ensure deepmimo is installed if running this locally
from deepmimo.exporters import sionna_exporter

save_folder = 'sionna_test_scen_abc_False'

sionna_exporter.sionna_exporter(scene, path_list, my_compute_path_params, save_folder)

#%%

scen_name_sionna = dm.convert(save_folder, overwrite=True)

#%%
scen_name_sionna = 'sionna_test_scen_abc_false'
dataset = dm.load(scen_name_sionna)
if type(dataset) == dm.MacroDataset: # multiple tx-rx pairs in the same scenario
    dataset[0].los.plot()
else:
    dataset.los.plot()
    
dataset.tx_pos

#%%

# Shape for synthetic = True (far-field)
# ch_params = dm.ChannelParameters()
# ch_params.bs_antenna.shape = (4, 1) # SHOULD MATCH THE NUMBER OF TX ANTENNAS
# dataset.compute_channels(ch_params)

# Shape for synthetic = True (near-field)
# ch_params = dm.ChannelParameters()
# ch_params.bs_antenna.shape = (4, 1) # SHOULD MATCH THE NUMBER OF TX ANTENNAS
# dataset.compute_channels_near_field(ch_params) # array response of near-field

# Shape for synthetic = False
ch_params = dm.ChannelParameters()
ch_params.bs_antenna.shape = (1, 1)
dataset.compute_channels(ch_params)
a = np.concatenate(dataset.channels, axis=1) # check axis

#%%


