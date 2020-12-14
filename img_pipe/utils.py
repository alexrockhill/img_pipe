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
            save_to_mat(op.join(surf_dir, f'{hemi}.{mesh_name}'), out_fname)


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


def point_to_polar(x, y, z):
    """Convert a point to zero-origin polar coordinates."""
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    return r, phi, theta


def polar_to_point(r, phi, theta):
    """Convert polar coordinates to a 3D point."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z
# assert polar_to_point(*point_to_polar(*loc2 - loc)) + loc == loc2


def template_error(loc, template, img):
    """Compute the correlation of a template with an image at a location."""
    loc = np.round(loc).astype(int)
    for i, c, s in zip(range(img.ndim), loc, template.shape):
        if c - s // 2 < 0 or c + s // 2 + s % 2 >= img.shape[i]:
            return np.inf  # no out of bounds comparisons
        img = \
            img[(slice(None),) * i + (slice(c - s // 2, c + s // 2 + s % 2),)]
    img = img.copy()  # avoid modifying the original
    img += img.min()
    if img.max() > 0:
        img /= img.max()
    return np.sum((img - template)**2) / template.size


def get_neighbors(loc, img, thresh, volume_indices):
    neighbors = set()
    for axis in range(len(loc)):
        for i in (-1, 1):
            n_loc = np.array(loc)
            n_loc[axis] += i
            n_loc = tuple(n_loc)
            if img[n_loc] > thresh and img[n_loc] < img[loc] and \
                    n_loc not in volume_indices:
                neighbors.add(n_loc)
    return neighbors


def peak_to_volume(loc, img, volume_max, volume_thresh=0.25):
    loc = tuple(loc)
    thresh = img[loc] * volume_thresh
    volume_indices = neighbors = set([loc])
    while neighbors and len(volume_indices) <= volume_max:
        next_neighbors = set()
        for n_loc in neighbors:
            this_next_neighbors = \
                get_neighbors(n_loc, img, thresh, volume_indices)
            volume_indices = volume_indices.union(this_next_neighbors)
            if len(volume_indices) > volume_max:
                break
            next_neighbors = next_neighbors.union(this_next_neighbors)
        neighbors = next_neighbors
    return volume_indices


def find_connected(seed, connect_mat, connections=None):
    """Recursively find all connected entries from a seed."""
    if connections is None:
        connections = set([seed])
    next_connections = \
        set(np.where(connect_mat[seed])[0]).difference(connections)
    connections = connections.union(next_connections)
    if len(next_connections) > 0:
        return connections.union(*[find_connected(
            idx, connect_mat, connections) for idx in next_connections])
    else:
        return set()


def get_devices(peaks, spacing_min, spacing_max, volumes, ct_data):
    dist_mat = scipy.spatial.distance_matrix(peaks, peaks)
    connect_mat = (dist_mat >= spacing_min) & (dist_mat <= spacing_max)
    devices = list()
    singles = list()
    scores = list()
    peaks_used = set()
    idx = 0
    while not all([i in peaks_used for i in range(peaks.shape[0])]):
        while idx in peaks_used:
            idx += 1
        if any(connect_mat[idx]):
            these_peaks = list(find_connected(idx, connect_mat))
            peaks_used = peaks_used.union(these_peaks)
            score = 0
            these_dists = list()
            for i, p0 in enumerate(these_peaks):
                for j, p1 in enumerate(these_peaks[i + 1:]):
                    if connect_mat[p0, p1]:
                        these_dists.append(dist_mat[p0, p1])
            score += np.std(these_dists)
            score += np.std([ct_data[tuple(peaks[i])] for i in these_peaks])
            score += np.std([len(volumes[i]) for i in these_peaks])
            if len(these_peaks) > 2:
                devices.append(these_peaks)
                scores.append(score / len(these_peaks))
            else:
                singles.append(these_peaks)
        else:
            peaks_used.add(idx)
            singles.append([idx])
    singles = [devices.pop(i) for i in range(len(devices))
               if np.isinf(scores[i])]
    devices = [devices[i] for i in np.argsort(scores)]
    scores = [score[i] for i in np.argsort(scores)]
    devices = \
        [devices[i] for i, test in enumerate(np.isfinite(scores)) if test]
    return devices, singles


def localize_devices_from_ct(volume_min=3, volume_max=36, volume_thresh=0.5,
                             spacing_min=3, spacing_max=7, intensity_min=0.65,
                             opacity=0.1, verbose=True):
    """Find groups of contacts from a CT image.

    Parameters
    ----------
    volume_min: int
        The minimum number of voxels within 'volume_thresh' proportion
        of that peak to use. Increase if small artifactual point are being
        found, decrease if small contacts are not being found.
    volume_max: int
        The minimum number of voxels within 'volume_thresh' proportion
        of that peak to use. Increase if large contacts are not being
        found, decrease if too many large artifacts are being found.
    volume_thresh: float
        The proportion of each peak to count as within the volume
        of the contact.
    spacing_min: float
        The minimum amount of space between contacts to use (in voxels).
        Increase if close clusters of artifactual points are being
        included as devices, decrease if devices are not being connected.
    spacing_max: float
        The maximum amount of space between contacts to use (in voxels).
        Increase if devices are not being connected, decrease if
        devices are improperly connected to each other.
    intensity_min: float
        The proportion of the greatest intensity-point that the contact
        maximum intensity voxel must be greater than.
    opacity: float
        The opacity of the CT data used to scaffold the visualization.
    verbose: bool
        Whether to print text updating on the status of the function.

    """
    from skimage.feature import peak_local_max
    ct_data = load_image_data('CT', 'rCT.nii', 'coreg_CT_MR',
                              reorient=True, verbose=verbose)
    if verbose:
        print('Finding local maxima on the CT...')
    peaks = peak_local_max(
        ct_data, threshold_abs=intensity_min * np.nanmax(ct_data))
    if verbose:
        print('Finding volumes around local maxima...')
    volumes = [peak_to_volume(peak, ct_data, volume_max, volume_thresh)
               for peak in peaks]
    n_voxels = np.array([len(vol) for vol in volumes])
    peaks = peaks[(n_voxels >= volume_min) & (n_voxels <= volume_max)]
    volumes = [vol for i, vol in enumerate(volumes) if
               len(vol) >= volume_min and len(vol) <= volume_max]
    assert len(volumes) == peaks.shape[0]
    if verbose:
        print(f'{peaks.shape[0]} voxels are local peaks with '
              f'volumes between {volume_min} and {volume_max}, '
              f'thresholded for volume at {volume_thresh} of that peak value')
        print('Determining devices from neighboring local maxima...')
    devices, singles = get_devices(peaks, spacing_min, spacing_max,
                                   volumes, ct_data)
    if verbose:
        print(f'{len(devices)} devices found')
    return devices


def gauss3D(amp, x0, y0, z0, sigma_x, sigma_y, sigma_z):
    return lambda x, y, z: amp * np.exp((-(x - x0) ** 2) / 2 * sigma_x**2)


def make_sphere(radius, size):
    size = np.round(size).astype(int)
    img = np.zeros((size,) * 3)
    indices = np.arange(-(size // 2), size // 2 + 1)
    if size % 2 == 0:
        indices = indices[1:] - 0.5
    dists = 0
    for i in range(img.ndim):
        dists = dists + indices.reshape(indices.shape + (1,) * i) ** 2
    dists = np.sqrt(dists)
    assert dists.shape == (size,) * img.ndim
    mask = (dists <= radius)
    if not mask.any():
        raise ValueError(f'No points withing radius {radius}')
    assert mask.shape == (size,) * img.ndim
    img[mask] = -(dists[mask] - dists[mask].max()) / dists[mask].max()
    return img


def get_radius(img, coords, r, r2=None):
    """Get the values within a radius of the ndimage."""
    assert coords.size == img.ndim
    remainders = coords % 1
    rr = np.ceil(r).astype(int) + 1
    indices = np.arange(-rr, rr + 1)
    dists = 0
    for i in range(img.ndim):
        dists = dists + (indices + remainders[i]).reshape(
            indices.shape + (1,) * i) ** 2
    dists = np.sqrt(dists)
    assert dists.shape == (rr * 2 + 1,) * img.ndim
    mask = (dists <= r) if r2 is None else ((r <= dists) & (dists <= r2))
    assert mask.shape == (rr * 2 + 1,) * img.ndim
    for i, c in zip(range(img.ndim), coords):
        in_bounds = ((indices + c) >= 0) & ((indices + c) < img.shape[i])
        mask = np.take(mask, np.arange(mask.shape[i])[in_bounds], axis=i)
        inds = indices[in_bounds] + int(c)
        img = img[(slice(None),) * i + (slice(inds[0], inds[-1] + 1),)]
    return img[mask]


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
        raise ValueError(f'Dimensions found were {img_data.shape} whereas'
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


def save_to_mat(fname, out_fname):
    """Take in a freesurfer surface and save it as a mat file."""
    vert, tri = nib.freesurfer.read_geometry(fname)
    scipy.io.savemat(out_fname, {'tri': tri, 'vert': vert})


def srf_to_surf(fname, out_fname):
    """Convert srf files to freesurfer surface files.

    ..depreciated Uses brainder bash/freesurfer script to tesselate
    surfaces, replaced by python marching cubes solution

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


def generate_orthogonal_colors(n_colors=20, n_candidate_points=10000,
                               n_color_samples=100000, unit=False, seed=11,
                               plot=False):
    '''Generate `orthogonal` colors that contrast highly with each other

    Note: this was called with default arguments and then the output was pasted
    to define `UNIQUE_COLORS` in img_pipe/config.py to save computation time
    '''
    # sample points in the first quadrent of a unit sphere
    np.random.seed(seed)

    if unit:  # make colors unit normalized (add up to one)
        points = abs(np.random.randn(n_candidate_points, 3))
        points /= np.sum(points, axis=1)[:, np.newaxis]
        points *= np.cbrt(np.random.random(n_candidate_points))[:, np.newaxis]
    else:
        points = np.random.random((n_candidate_points, 3))
    dist_mat = scipy.spatial.distance_matrix(points, points)
    counter = 0
    best_score = best_color_indices = best_min_dists = None
    while counter < n_color_samples:
        color_indices = np.random.randint(0, n_candidate_points, n_colors)
        min_dists = [dist_mat[idx, color_indices[color_indices != idx]].min()
                     for idx in color_indices]
        this_best_score = np.mean(min_dists) / np.std(min_dists)
        if best_score is None or best_score < this_best_score:
            best_score = this_best_score
            best_min_dists = min_dists
            best_color_indices = color_indices
        counter += 1

    unique_colors = [tuple(np.round(1 - points[i], 2)) for i in
                     best_color_indices[np.argsort(best_min_dists)[::-1]]]

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(20):
            ax.scatter(i, i, color=unique_colors[i], s=1000)
        fig.show()

    return unique_colors
