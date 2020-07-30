import os.path as op
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from img_pipe.utils import check_fs_vars, check_file

import ctmr_brain_plot


def get_elecs_anat(region):
    base_path = check_fs_vars()
    tdt_fname = check_file(op.join(base_path, 'elecs', 'TDT_elecs_all.mat'))
    tdt = scipy.io.loadmat(tdt_fname)
    return tdt['elecmatrix'][np.where(tdt['anatomy'][:, 3] == region)[0], :]


def ctmr_plot(hemi, elecs, weights=None, interactive=False):
    base_path = check_fs_vars()
    hemi_data_fname = check_file(op.join(base_path, 'meshes',
                                 '{}_pial_trivert.mat'.format(hemi)))
    hemi_array = scipy.io.loadmat(hemi_data_fname)
    if weights is None:
        weights = np.ones((elecs.shape[0])) * -1.
    mesh, mlab = ctmr_brain_plot.ctmr_gauss_plot(hemi_array['tri'],
                                                 hemi_array['vert'],
                                                 elecs=elecs, weights=weights,
                                                 color=(0.8, 0.8, 0.8),
                                                 cmap='RdBu')

    mesh.actor.property.opacity = 1.0  # Make brain semi-transparent

    # View from the side
    if hemi == 'lh':
        azimuth = 180
    elif hemi == 'rh':
        azimuth = 0
    mlab.view(azimuth, elevation=90)
    arr = mlab.screenshot(antialiased=True)
    plt.figure(figsize=(20, 10))
    plt.imshow(arr, aspect='equal')
    plt.axis('off')
    plt.show()
    if interactive:
        mlab.show()
    else:
        mlab.close()
    return mesh, mlab
