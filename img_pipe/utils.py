"""Utility and helper functions for img_pipe."""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD (3-clause)

import os
import os.path as op
import numpy as np

import scipy
from skimage import measure
import nibabel as nib
import mne

from img_pipe.config import VOXEL_SIZES, CORTICAL_SURFACES


def list_rois(atlas='desikan-killiany', template=None):
    """Lists the regions of interest available for plotting.

    Parameters
    ----------
    atlas: str
        The segementation atlas to use; 'desikan-killiany' (default),
        'DKT' or 'destrieux'.
    template: str
        The common reference brain template to use, defaults to the
        individual subject. May be 'V1_average', 'cvs_avg35',
        'cvs_avg35_inMNI152', 'fsaverage', 'fsaverage3', 'fsaverage4',
        'fsaverage5', 'fsaverage6' or 'fsaverage_sym'.

    """
    base_path = check_fs_vars()
    if template is None:
        label_dir = check_dir(op.join(base_path, 'label', atlas),
                              'img_pipe.label')
    else:
        label_dir = check_dir(op.join(
            os.environ['FREESURFER_HOME'], 'subjects', template, 'label',
            atlas), 'img_pipe.warp')
    return list(CORTICAL_SURFACES.keys()) + os.listdir(label_dir)


def export_labels(overwrite=False, verbose=True):
    """Converts freesurfer surfaces to mat files.

    Parameters
    ----------
    overwrite : bool
        Whether to overwrite the target filepath.
    verbose : bool
        Whether to print text updating on the status of the function.
    """
    base_path = check_fs_vars()
    surf_dir = check_dir(op.join(base_path, 'surf'), 'recon')
    mesh_dir = make_dir(op.join(base_path, 'meshes'))
    # loop through types of mesh for plotting
    for mesh_name in ('pial', 'white', 'inflated'):
        # loop through hemispheres for this mesh, create one .mat file for each
        for hemi in ('lh', 'rh'):
            if verbose:
                print(f'Making {hemi} {mesh_name} mesh')
            out_fname = op.join(mesh_dir, '{}_{}_{}.mat'.format(
                os.environ['SUBJECT'], hemi, mesh_name))
            if op.isfile(out_fname) and not overwrite:
                raise ValueError(f'File {out_fname} exists and '
                                 'overwrite is False')
            save2mat(op.join(surf_dir, f'{hemi}.{mesh_name}'), out_fname)


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


def load_raw(verbose=True):
    """Load the intracranial electrophysiology data file."""
    if verbose:
        print('Loading an electrophysiology file')
    fname = get_ieeg_fnames(return_first=True, verbose=verbose)
    ext = op.splitext(fname)[-1]
    if ext == '.fif':
        raw = mne.io.read_raw_fif(fname, preload=False)
    elif ext == '.edf':
        raw = mne.io.read_raw_edf(fname, preload=False)
    elif ext == '.bdf':
        raw = mne.io.read_raw_bdf(fname, preload=False)
    elif ext == '.vhdr':
        raw = mne.io.read_raw_brainvision(fname, preload=False)
    elif ext == '.set':
        raw = mne.io.read_raw_eeglab(fname, preload=False)
    ch_indices = mne.pick_types(raw.info, meg=False, eeg=True,
                                ecog=True, seeg=True)
    if not len(ch_indices) > 0:
        raise ValueError(f'No eeg, ecog or seeg channels found in {fname}, '
                         'check that the data is correctly typed')
    return raw


def load_electrode_names(verbose=True):
    """Loads the names of electrodes from the electrophysiology data file."""
    if verbose:
        print('Loading electrode names')
    raw = load_raw(verbose=verbose)
    return [ch for ch in raw.ch_names if ch not in ('Event', 'STI 014')]


def load_electrodes(verbose=True):
    """Load the registered electrodes."""
    if verbose:
        print('Loading electrode matrix')
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


def save_electrodes(elec_matrix, verbose=True):
    """Save the location of the electrodes."""
    if verbose:
        print('Saving electrode positions')
    base_path = check_fs_vars()
    elec_fname = op.join(base_path, 'elecs', 'electrodes.tsv')
    with open(elec_fname, 'w') as fid:
        fid.write('\t'.join(['name', 'R', 'A', 'S', 'group', 'label']) + '\n')
        for name in elec_matrix:  # sort as given
            x, y, z, group, label = elec_matrix[name]
            fid.write('\t'.join(np.array(
                [name, x, y, z, int(group), label]).astype(str)) + '\n')


def load_image_data(dirname, basename, function='img_pipe.recon',
                    reorient=False, verbose=True):
    """Load data from a 3D image file (e.g. CT, MR)."""
    if verbose:
        print(f'Loading {basename}')
    base_path = check_fs_vars()
    ext = op.splitext(basename)[-1]
    fname = check_file(op.join(base_path, dirname, basename), function)
    if ext == 'mgz':
        img = nib.freesurfer.load(fname)
    else:
        img = nib.load(fname)
    if reorient:
        img_data = nib.orientations.apply_orientation(
            img.get_fdata(), nib.orientations.axcodes2ornt(
                nib.orientations.aff2axcodes(img.affine)))
    else:
        img_data = img.get_fdata()
    if not np.array_equal(np.array(img_data.shape, dtype=int), VOXEL_SIZES):
        raise ValueError(f'MRI dimensions found {img_data.shape} '
                         f'expected dimensions were {VOXEL_SIZES}, '
                         'check recon and contact developers')
    return img_data


def get_vox_to_ras(inverse=False):
    """Covert coordinates in voxel space to right-anterior-superior (RAS)."""
    base_path = check_fs_vars()
    t1w = nib.load(check_file(op.join(base_path, 'mri', 'T1.mgz'),
                              'img_pipe.recon'))
    t1w = nib.Nifti1Image(t1w.dataobj, t1w.affine)
    t1w.header['xyzt_units'] = np.array(10, dtype='uint8')
    t1_mgh = nib.MGHImage(t1w.dataobj, t1w.affine)
    vox_to_ras = t1_mgh.header.get_vox2ras_tkr()
    ras_to_vox = scipy.linalg.inv(vox_to_ras)
    return vox_to_ras, ras_to_vox


def apply_trans(trans, data):
    """Use a transform to change coordinate spaces."""
    return mne.transforms.apply_trans(trans, data, move=True)


def aseg_to_surf(out_fname, aseg, idx, trans, sigma=1, verbose=True):
    """Creates a mesh surface from the voxel segementation data.

    Adapted from https://github.com/mmvt/mmvt/blob/master/src/
    misc/create_subcortical_surf.py
    """
    img = np.where(aseg == idx, 10, 255).astype(float)
    img_smooth = scipy.ndimage.gaussian_filter(img, sigma=sigma)
    if img_smooth.min() > 100:
        if verbose:
            print(f'{out_fname} volume too small with {sigma} smoothing, '
                  'if you need this area, decrease sigma')
        return
    # find a surface less than level 100 with the area of interest at 10
    # and the background at 255
    vert, tri, _, _ = measure.marching_cubes(img_smooth, level=100)
    # voxels to ras coordinates
    vert = mne.transforms.apply_trans(trans, vert, move=True)
    nib.freesurfer.io.write_geometry(out_fname, vert, tri)


def get_vert_labels(verbose=True):
    """Load the freesurfer vertex labels."""
    if verbose:
        print('Loading vertex labels')
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


def srf2surf(fname, out_fname):
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
    """Get the corresponance between numbers and freesurfer areas."""
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
    """Get the colors of freesurfer areas."""
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
