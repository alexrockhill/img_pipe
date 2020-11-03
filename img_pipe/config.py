"""Configuration variables for img_pipe."""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD (3-clause)

import numpy as np

VOXEL_SIZES = np.array([256, 256, 256])
ELEC_PLOT_SIZE = np.array([1024, 1024])
ZOOM_STEP_SIZE = 5
CT_MIN_VAL = 1000
MAX_N_GROUPS = 17

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
