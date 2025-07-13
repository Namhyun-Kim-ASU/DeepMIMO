import deepmimo as dm
import numpy as np
import matplotlib.pyplot as plt

save_plots = True
save_dir = 'path_trim_plots'

dataset = dm.load('asu_campus_3p5')

# Calculate total number of paths for reference
n_active_ue = len(dataset.get_active_idxs())
n_active_paths = dataset.num_paths.sum()

# Get power limits from original dataset for consistency
pwr_lims = [np.nanmin(dataset.power), np.nanmax(dataset.power)]

# Define combinations of path types and depths to analyze
# Each tuple is (path_types, max_depth)
# None for path_types means all types allowed
# None for max_depth means no depth filtering
path_combinations = [
    # First few combinations without depth filtering
    (['LoS'], None),
    (['LoS', 'R'], None),
    (['LoS', 'R', 'D'], None),
    (['LoS', 'R', 'S'], None),
    
    # # Single interaction types with depth filtering
    (['R'], None),
    (['D'], None),
    (['S'], None),
    
    # # Multiple interactions with depth filtering
    (['R'], None),
    (['R', 'D'], None),
    (['R', 'S'], None),
    
    # All types with different depths
    (None, 0),
    (None, 1),
    (None, 2),
    (None, 3),
    (None, 4),
    (None, 5),
    (None, 6),

    # Joint path type and depth filtering
    (['R'], 1),
    (['R'], 2),
    (['R'], 3),
    (['R'], 4),
    (['R'], 5), # 5 reflections in ASU
    
    (['D'], 1), # only one diffraction in ASU
    
    (['R', 'S'], 1),
    (['R', 'S'], 2),
    (['R', 'S'], 3),
    (['R', 'S'], 4),
    (['R', 'S'], 5),
    (['R', 'S'], 6), # 1 scattering event after reflections ASU

    (['R', 'D'], 1),
    (['R', 'D'], 2),
    (['R', 'D'], 3),
    (['R', 'D'], 4),
    (['R', 'D'], 5),
    (['R', 'D'], 6),

    (['R', 'D', 'S'], 6), # all except LoS
    (['LoS', 'R', 'D', 'S'], 6), # 100% of path coverage
]

# Process each combination
for combo_idx, (path_types, max_depth) in enumerate(path_combinations):
    # Start with original dataset
    trimmed_dataset = dataset
    
    # Apply path type filtering if specified
    type_title = 'All types'
    if path_types is not None:
        type_title = ' + '.join(path_types)
        print(f'trimming by path type: {" + ".join(path_types)}')
        trimmed_dataset = trimmed_dataset.trim_by_path_type(path_types)
    
    # Apply depth filtering if specified
    depth_title = 'All depths'
    if max_depth is not None:
        depth_title = f'depth≤{max_depth}'
        print(f'trimming by path depth: {max_depth}')
        trimmed_dataset = trimmed_dataset.trim_by_path_depth(max_depth)
    
    # Create title with trim type
    title = f'{type_title} | {depth_title}'
        
    # Calculate coverage percentage
    coverage = len(trimmed_dataset.get_active_idxs()) / n_active_ue
    path_coverage = trimmed_dataset.num_paths.sum() / n_active_paths
    
    plt_args = {
        'dpi': 200,
        'title': f'{title} ({coverage:.2%}, paths:{path_coverage:.2%})',
        'lims': pwr_lims,
    }
    # Plot power with coverage percentage in title and consistent limits
    trimmed_dataset.power.plot(**plt_args)
    
    if save_plots:
        path_type_str = '-'.join(path_types) if path_types else 'all-types'
        depth_str = str(max_depth) if max_depth is not None else 'all-depths'
        exp_str = f'{combo_idx:03d}_{path_type_str}_{depth_str}'
        save_path = f'{save_dir}/pwr_exp_{exp_str}.png'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path)

    plt.show()
    # break

    # FUTURE: Compute channels & save matrix for topological analysis

    # FUN FACT: trimming by depth is much faster than trimming by type.