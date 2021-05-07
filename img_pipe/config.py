"""Configuration variables for img_pipe."""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD (3-clause)

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

VOXEL_SIZES = np.array([256, 256, 256])
IMG_RANGES = [[0, VOXEL_SIZES[1], 0, VOXEL_SIZES[2]],
              [0, VOXEL_SIZES[0], 0, VOXEL_SIZES[2]],
              [0, VOXEL_SIZES[0], 0, VOXEL_SIZES[1]]]
IMG_LABELS = [['Inferior', 'Posterior'],
              ['Inferior', 'Left'],
              ['Posterior', 'Left']]
ELEC_PLOT_SIZE = np.array([1024, 1024])
ZOOM_STEP_SIZE = 5
CT_MIN_VAL = 1000
MAX_N_GROUPS = 25

UNIQUE_COLORS = [(0.1, 0.42, 0.43), (0.9, 0.34, 0.62), (0.47, 0.51, 0.3),
                 (0.47, 0.55, 0.99), (0.79, 0.68, 0.06), (0.34, 0.74, 0.05),
                 (0.58, 0.87, 0.13), (0.86, 0.98, 0.4), (0.92, 0.91, 0.66),
                 (0.77, 0.38, 0.34), (0.9, 0.37, 0.1), (0.2, 0.62, 0.9),
                 (0.22, 0.65, 0.64), (0.14, 0.94, 0.8), (0.34, 0.31, 0.68),
                 (0.59, 0.28, 0.74), (0.46, 0.19, 0.94), (0.37, 0.93, 0.7),
                 (0.56, 0.86, 0.55), (0.67, 0.69, 0.44)]
N_COLORS = len(UNIQUE_COLORS)
ELECTRODE_CMAP = LinearSegmentedColormap.from_list(
    'elec_colors', UNIQUE_COLORS, N=N_COLORS)


ATLAS_DICT = {'desikan-killiany': 'aparc',
              'DKT': 'aparc.DKTatlas',
              'destrieux': 'aparc.a2009s'}

TEMPLATES = ['V1_average', 'cvs_avg35', 'cvs_avg35_inMNI152', 'fsaverage',
             'fsaverage3', 'fsaverage4', 'fsaverage5', 'fsaverage6',
             'fsaverage_sym']

HEMI_DICT = {'Left': 'lh', 'Right': 'rh'}

CORTICAL_SURFACES = {f'{hemi}-{roi}': f'{HEMI_DICT[hemi]}.{roi.lower()}'
                     for hemi in ('Left', 'Right')
                     for roi in ('Pial', 'Inflated', 'White')}

SUBCORTICAL_INDICES = [4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26,
                       28, 43, 44, 49, 50, 51, 52, 53, 54, 58, 60]
