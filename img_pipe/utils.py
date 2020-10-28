"""Utility and helper functions for img_pipe."""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD (3-clause)

import os
import os.path as op
import numpy as np
import scipy
from scipy.io import loadmat, savemat
import nibabel as nib


def check_file(my_file, function=None, instructions=None):
    if instructions is None:
        instructions = f'run {function} to make it'
    if not op.isfile(my_file):
        raise ValueError(f'File {my_file} does not exist, {instructions}')
    return my_file


def check_dir(my_dir, function=None, instructions=None):
    """Make a directory if it doesn't exist"""
    if instructions is None:
        instructions = f'run {function} to make it'
    if not op.isdir(my_dir):
        raise ValueError(f'Directory {my_dir} does not exist, {instructions}')
    return my_dir


def make_dir(my_dir):
    """Make a directory if it doesn't exist"""
    if not op.isdir(my_dir):
        os.makedirs(my_dir)
    return my_dir


def check_fs_vars():
    """Checks whether SUBJECT and SUBJECTS_DIR are in os.environ"""
    error_str = ('Freesurfer parameter {} not set, this value must be '
                 'set before continuing')
    for var in ('SUBJECT', 'SUBJECTS_DIR'):
        if var not in os.environ:
            raise ValueError(error_str.format(var))
    return op.join(os.environ['SUBJECTS_DIR'], os.environ['SUBJECT'])


def check_hemi(hemi):
    if hemi not in ('lh', 'rh', 'both'):
        raise ValueError(f'Unexpected hemi argument {hemi}')
    return ('lh', 'rh') if hemi == 'both' else (hemi,)


def get_azimuth(rois):
    hemis = np.unique([roi.hemi for roi in rois])
    if 'both' in hemis or ('lh' in hemis and 'rh' in hemis):
        return 90
    return 180 if 'lh' in hemis else 0


def get_ieeg_fnames(return_first=False, verbose=True):
    base_path = check_fs_vars()
    ieeg_fnames = [op.join(base_path, 'ieeg', fname) for fname in
                   os.listdir(op.join(base_path, 'ieeg'))
                   if op.splitext(fname)[-1][1:] in
                   ('fif', 'edf', 'bdf', 'vhdr', 'set')]
    if len(ieeg_fnames) > 1:
        if verbose:
            print('Warning {} data files, {}, found in ieeg, using names'
                  'from the first one'.format(len(ieeg_fnames), ieeg_fnames))
    elif len(ieeg_fnames) == 0:
        if len(os.listdir(op.join(base_path, 'ieeg'))) > 0:
            for fname in os.listdir(op.join(base_path, 'ieeg')):
                raise ValueError('Extension {} not recognized, options are'
                                 'fif, edf, bdf, vhdr (brainvision) and set '
                                 '(eeglab)'.format(op.splitext(fname)[-1]))
        return None
    if return_first:
        return ieeg_fnames[0]
    else:
        return ieeg_fnames


def load_electrode_names():
    base_path = check_fs_vars()
    elec_name_fname = op.join(base_path, 'elecs', 'electrode_names.tsv')
    elec_names = list()
    if op.isfile(elec_name_fname):
        with open(elec_name_fname, 'r') as fid:
            headers = fid.readline().rstrip().split('\t')
            name_idx = headers.index('name')
            for line in fid:
                elec_names.append(line.rstrip().split('\t')[name_idx])
    return elec_names


def load_electrodes():
    base_path = check_fs_vars()
    elec_fname = op.join(base_path, 'elecs', 'electrodes.tsv')
    elec_matrix = dict()
    if not op.isfile(elec_fname):
        return elec_matrix
    with open(elec_fname, 'r') as fid:
        header = fid.readline()  # for header
        assert header.rstrip().split('\t') == ['name', 'R', 'A', 'S',
                                               'group', 'label']
        for line in fid:
            name, R, A, S, group, label = line.rstrip().split('\t')
            elec_data = np.array([R, A, S]).astype(float).tolist()
            elec_data += [int(group), label]
            elec_matrix[name] = elec_data
    return elec_matrix


def save_electrodes(elec_matrix):
    base_path = check_fs_vars()
    elec_fname = op.join(base_path, 'elecs', 'electrodes.tsv')
    with open(elec_fname, 'w') as fid:
        fid.write('\t'.join(['name', 'R', 'A', 'S', 'group', 'label']) + '\n')
        for name in elec_matrix:  # sort as given
            x, y, z, group, label = elec_matrix[name]
            fid.write('\t'.join(np.array(
                [name, x, y, z, int(group), label]).astype(str)) + '\n')


def load_image_data(dirname, basename, function='recon-all'):
    base_path = check_fs_vars()
    ext = op.splitext(basename)[-1]
    fname = check_file(op.join(base_path, dirname, basename), function)
    if ext == 'mgz':
        img = nib.freesurfer.load(fname)
    else:
        img = nib.load(fname)
    codes = nib.orientations.axcodes2ornt(
        nib.orientations.aff2axcodes(img.affine))
    img_data = nib.orientations.apply_orientation(img.get_fdata(), codes)
    return img_data


def get_vert_labels():
    base_path = check_fs_vars()
    gyri_labels_dir = check_dir(op.join(base_path, 'label', 'gyri'))
    label_fnames = [op.join(gyri_labels_dir, f) for f in
                    os.listdir(gyri_labels_dir) if f.endswith('label')]
    vert_labels = dict()
    for label_fname in label_fnames:
        label_name = label_fname.split('.')[1].strip()
        label_data = np.loadtxt(label_fname, skiprows=2)
        for v in label_data[:, 0].astype(int):
            vert_labels[v] = label_name
    return vert_labels


def save2mat(fname, out_fname):
    """Take in a freesurfer surface and save it as a mat file."""
    vert, tri = nib.freesurfer.read_geometry(fname)
    scipy.io.savemat(out_fname, {'tri': tri, 'vert': vert})


def get_surf(hemi, roi, template=None):
    """ Utility for loading the pial surface for a given hemisphere.

    Parameters
    ----------
    hem : str
        Hemisphere for the surface. If blank, defaults to 'both'.
        Otherwise, use 'lh' or 'rh' for the left or right hemisphere
        respectively.
    roi : str
        The region of interest to load.  Should be a Mesh that exists
        in the subject's Meshes directory, called [roi]_trivert.mat
    template : str, optional
        Name of the template to use if plotting electrodes on an
        atlas brain. e.g. 'cvs_avg35_inMNI152'

    Returns
    -------
    cortex : dict
        Dictionary containing 'tri' and 'vert' for the loaded region
        of interest mesh.
    """
    base_path = check_fs_vars()
    if template is None:
        mesh_dir = check_dir(op.join(base_path, 'meshes'), 'label')
    else:
        mesh_dir = check_dir(op.join(os.environ['SUBJECTS_DIR'], template),
                             'label')
    vert_data_fname = check_file(
        op.join(mesh_dir, f'{hemi}_{roi}_trivert.mat'), 'label')
    return loadmat(vert_data_fname)


def subcort2surf(fname, out_fname):
    """Convert srf files to freesurfer surface files.

    Parameters
    ----------
    fname : str
        The directory to save out to
    out_fname : str
        Name of the subcortical mesh ascii file (e.g. aseg_058.asc).
    """
    # use freesurfer mris_convert to get ascii subcortical surface
    with open(fname, 'r') as fid:
        fid.readline()  # get rid of comments in header
        subcort_mat = [line.rstrip() for line in fid]
    # extract inds for vert and tri
    subcort_inds = [int(i) for i in subcort_mat.pop(0).split(' ')]
    subcort_vert = [item.strip(' 0')
                    for item in subcort_mat[:subcort_inds[0]]]
    subcort_vert = [item.split('  ')
                    for item in subcort_vert]  # seperate strings
    # Convert into an array of floats
    subcort_vert = np.array(np.vstack((subcort_vert)), dtype=np.float)
    # get rows for triangles only, strip 0 column, and split into seperate
    # strings
    subcort_tri = [item[:-2] for item in subcort_mat[subcort_inds[0] + 1:]]
    subcort_tri = [item.split(' ')
                   for item in subcort_tri]  # seperate strings
    subcort_tri = np.array(np.vstack((subcort_tri)), dtype=np.int)
    nib.freesurfer.io.write_geometry(out_fname, subcort_vert, subcort_tri)


def get_fs_labels():
    if 'FREESURFER_HOME' not in os.environ:
        raise ValueError('FREESURFER_HOME not in environment, '
                         'freesurfer was not sourced, source '
                         'to continue')
    label_fname = check_file(op.join(os.environ['FREESURFER_HOME'],
                                     'FreeSurferColorLUT.txt'),
                             instructions='check your freesurfer installation')
    number_dict = dict()
    with open(label_fname, 'r') as fid:
        for line in fid:
            line = line.rstrip()
            if line and line[0] != '#':  # exclude empty lines and comments
                n, name, r, g, b, a = line.split()
                number_dict[int(n)] = name
                number_dict[name] = int(n)
    return number_dict


def get_fs_colors():
    if 'FREESURFER_HOME' not in os.environ:
        raise ValueError('FREESURFER_HOME not in environment, '
                         'freesurfer was not sourced, source '
                         'to continue')
    label_fname = check_file(op.join(os.environ['FREESURFER_HOME'],
                                     'FreeSurferColorLUT.txt'),
                             instructions='check your freesurfer installation')
    color_dict = dict()
    with open(label_fname, 'r') as fid:
        for line in fid:
            line = line.rstrip()
            if line and line[0] != '#':  # exclude empty lines and comments
                _, name, r, g, b, _ = line.split()
                color_dict[name] = (int(r), int(g), int(b))
    return color_dict


def shorten_name(name):
    return name.replace('-', '').replace('Left', 'l').replace('Right', 'r')


# For animations, from pycortex
def linear(x, y, m):
    (1. - m) * x + m * y


def smooth_step(x, y, m):
    linear(x, y, 3 * m ** 2 - 2 * m ** 3)


def smoother_step(x, y, m):
    linear(x, y, 6 * m ** 5 - 15 * m ** 4 + 10 * m ** 3)


mixes = dict(linear=linear, smooth_step=smooth_step,
             smoother_step=smoother_step)
