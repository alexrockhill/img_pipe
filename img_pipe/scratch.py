"""Unused but useful backup functions for img_pipe."""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD (3-clause)

import os
import os.path as op
from img_pipe.utils import check_file, check_fs_vars


ROSA_COLORS = [(0.65, 0.13, 0.37), (0.37, 0.75, 0.99), (0.58, 0.01, 0.61),
               (1.0, 0.4, 0.0), (0.01, 0.8, 1.0), (0.0, 1.0, 0.17),
               (0.99, 0.0, 0.82), (1.0, 1.0, 0.78), (0.77, 0.56, 1.0),
               (0.79, 0.59, 0.26), (1.0, 0.55, 0.54), (0.34, 0.99, 0.79),
               (0.53, 0.57, 1.0), (1.0, 0.81, 0.62), (0.01, 0.0, 0.95),
               (0.0, 0.64, 0.0), (0.97, 1.0, 0.54)]


def check_fsl():
    if 'FSLDIR' not in os.environ:
        raise ImportError('FSL not installed, please install to continue. '
                          'CT can be registered with MR by hand without '
                          'FSL but this decreases reproducibility and '
                          'likely accuracy and is not recommended')
    if os.system('flirt -version'):
        raise ImportError('flirt not intitialized, likely because fsl '
                          'has not been added to the path. Add $FSLDIR/bin '
                          'to the path to solve this issue e.g. '
                          'export PATH=$FSLDIR/bin${PATH:+:${PATH}}')


def coreg_CT_MR(overwrite=False, verbose=True):
    '''Coregistrs the anatomical MR with the CT using fsl.

    Parameters are arguments for
    nipy.algorithms.registration.histogram_registration.HistogramRegistration
    (for more information, see help for
    nipy.algorithms.registration.optimizer).

    Parameters
    ----------
    overwrite : bool
        Whether to overwrite the target filepath
    verbose : bool
        Whether to print text updating on the status of the function.

    Jenkinson, M., Bannister, P., Brady, J. M. and Smith, S. M.
    Improved Optimisation for the Robust and Accurate Linear Registration and
    Motion Correction of Brain Images. NeuroImage, 17(2), 825-841, 2002.

    Jenkinson, M. and Smith, S. M. A Global Optimisation Method for Robust
    Affine Registration of Brain Images.
    Medical Image Analysis, 5(2), 143-156, 2001.

    Greve, D.N. and Fischl, B. Accurate and robust brain image alignment
    using boundary-based registration. NeuroImage, 48(1):63-72, 2009.
    '''

    base_path = check_fs_vars()
    check_fsl()

    in_fname = check_file(op.join(base_path, 'CT', 'CT.nii'))
    brainmask_fname_mgz = check_file(op.join(base_path, 'mri',
                                             'brainmask.mgz'))
    brainmask_fname = op.splitext(brainmask_fname_mgz)[0] + '.nii'
    ref_fname = check_file(op.join(base_path, 'mri', 'orig', 'T1.nii'))
    out_fname = op.join(base_path, 'CT', 'rCT.nii')
    omat_fname = op.join(base_path, 'CT', 'ct2brainmask.mat')
    omat2_fname = op.join(base_path, 'CT', 'brainmask2T1.mat')

    if op.isfile(out_fname) and not overwrite:
        raise ValueError(f'{out_fname} exists, use `overwrite=True` to '
                         'overwrite')

    if verbose:
        print(f'Computing registration from {in_fname} to {ref_fname}'
              '\n\nIf this is taking too long/doesn\'t work you may want '
              'you may want to translate the CT to be over the MR in freeview '
              'using `cd $SUBJECTS_DIR; freeview {$SUBJECT}/mri/orig/T1.nii '
              '{$SUBJECT}/CT/CT.nii adjusting the opacity and using tools>'
              'Transform Volume and translate and scale to put the CT '
              'roughly in the right place before saving over the original CT')

    # convert brainmask to nii so that it can be read in by fsl
    os.system(f'mri_convert {brainmask_fname_mgz} {brainmask_fname}')

    # register CT to brainmask for initial position as in iELVis
    os.system(f'flirt -in {in_fname} -ref {brainmask_fname} '
              f'-omat {omat_fname} -interp trilinear -cost normmi -dof 6 '
              '-searchcost normmi -usesqform '
              '-searchrx -180 180 -searchry -180 180 -searchrz -180 180 '
              f'-verbose {int(verbose)}')

    # use initial position to register ct to mri
    os.system(f'flirt -in {in_fname} -ref {ref_fname} -omat {omat2_fname}'
              f'-out {out_fname} -init {omat_fname} -interp trilinear '
              '-cost normmi -usesqform -dof 6 -searchcost normmi '
              f'-verbose {int(verbose)}')


or hemi in ('lh', 'rh'):
    if verbose:
        print(f'Making {atlas} labels for {hemi}')
    label_dir = make_dir(op.join(base_path, 'label', atlas))
    # make labels
    run(['mri_annotation2label', '--annotation', seg,
         '--subject', os.environ['SUBJECT'], '--hemi', hemi,
         '--surface', 'pial', '--outdir', label_dir])
    vol_dir = make_dir(op.join(base_path, 'label', f'{atlas}_vol'))
    tmp_dir = make_dir(op.join(base_path, 'label', f'{atlas}_tmp'))
    for label in os.listdir(label_dir):
        # make volume
        vol_fname = op.join(tmp_dir, f'{label}.mgz')
        filled_fname = op.join(tmp_dir, f'filled_{label}.mgz')
        cropped_fname = op.join(vol_dir, f'{label}.mgz')
        run(['mri_label2vol', '--label', op.join(label_dir, label),
             '--subject', os.environ['SUBJECT'], '--hemi', hemi,
             '--identity', '--temp', T1_fname, '--o', vol_fname])
        # fill volume
        run(['mri_binarize', '--dilate', '1', '--erode', '1',
             '--i', vol_fname, '--o', filled_fname, '--min', '1'])
        # crop volume with ribbon
        run(['mris_calc', '-o', cropped_fname, filled_fname, 'mul',
             ribbon_fnames[hemi]])

    # make subcortical labels
    surf_dir = check_dir(op.join(base_path, 'surf'), 'img_pipe.recon')
    # get the names of the numbered regions
    number_dict = get_fs_labels()

    # tessellate all subjects freesurfer subcortical segmentations
    if verbose:
        print('Tesselating freesurfer subcortical segmentations '
              'from aseg using aseg2srf...')
    if run([op.join(op.dirname(__file__), 'aseg2srf.sh'), '-s',
            os.environ['SUBJECT'], '-l',
            ' '.join([str(i) for i in SUBCORTICAL_INDICES])]).returncode:
        raise RuntimeError('error in aseg2srf.sh')

    ascii_dir = check_dir(op.join(base_path, 'ascii'),
                          instructions='aseg2srf error check with developers')

    if verbose:
        print('Converting all ascii segmentations surface geometry')
    for i in SUBCORTICAL_INDICES:
        fname = check_file(op.join(ascii_dir, 'aseg_{:03d}.srf'.format(i)),
                           instructions='`aseg2surf` error, please report')
        out_fname = op.join(surf_dir, f'sc.{surf_name(number_dict[i])}')
        srf2surf(fname, out_fname)


def plot_point(x, y, z, size, img):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.imshow(img[x, y - size: y + size + 1, z - size:z + size + 1].T)
    ax0.invert_yaxis()
    ax1.imshow(img[x - size: x + size + 1, y, z - size: z + size + 1].T)
    ax1.invert_yaxis()
    ax2.imshow(img[x - size: x + size + 1, y - size: y + size + 1, z].T)
    ax2.invert_yaxis()
    fig.show()


def check_match(x, y, z, t):
    sx, sy, sz = np.array(t.shape) // 2
    img = ct_data[x - sx: x + sx + 1]
    img = img[:, y - sy: y + sy + 1]
    img = img[:, :, z - sz: z + sz + 1]
    return np.correlate(img.flatten(), t.flatten())


electrode_template = ct_data[x - size: x + size + 1, y - size: y + size + 1, z - size: z + size + 1]
ij = np.unravel_index(np.argmax(electrode_match), electrode_match.shape)


electrode_match = feature.match_template(ct_data, electrode_template,
                                             pad_input=True)
electrode_match[:2 * electrode_template.shape[0]] = 0
electrode_match[-2 * electrode_template.shape[0]:] = 0
electrode_match[:, :2 * electrode_template.shape[1]] = 0
electrode_match[:, -2 * electrode_template.shape[1]:] = 0
electrode_match[:, :, :2 * electrode_template.shape[2]] = 0
electrode_match[:, :, -2 * electrode_template.shape[2]:] = 0
renderer = mne.viz.backends.renderer.create_3d_figure(
    size=(1200, 900), bgcolor='w', scene=False)
vert, tri, _, _ = measure.marching_cubes(ct_data)
renderer.mesh(*vert.T, triangles=tri, color=(0.8,) * 3, opacity=0.1)

b_ct_data = ct_data > np.quantile(ct_data.flatten(), threshold)
# flood fill features (generate_binary_structure includes diagonals)
labeled_ct, n_features = \
    label(b_ct_data, structure=generate_binary_structure(3, 3))
labels, counts = np.unique(labeled_ct, return_counts=True)
my_labels = labels[(10 < counts) & (counts < ct_data.size / 100)]
renderer = mne.viz.backends.renderer.create_3d_figure(
    size=(1200, 900), bgcolor='w', scene=False)
for my_label in my_labels:
    vert, tri, _, _ = measure.marching_cubes(labeled_ct == my_label)
    renderer.mesh(*vert.T, triangles=tri, color=(0.8,) * 3, opacity=0.1)

ct_shape = np.array(ct_data.shape)

loc = np.array([112, 145, 179])  # connected electrode
loc = np.array([108, 146, 162])  # lower down than above
loc = np.array([111, 146, 172])  # between two above
loc = np.array([155, 155, 186])  # skull peak
loc = np.array([92, 92, 122])    # lumpy electrode
loc = np.array([81, 89, 122])    # missing electrode


def get_device_data(contacts):  # dist_mat, connect_mat, ct_data, peaks, volumes):
    these_dists = list()
    for i, p0 in enumerate(contacts):
        for j, p1 in enumerate(contacts[i + 1:]):
            if connect_mat[p0, p1]:
                these_dists.append(dist_mat[p0, p1])
    print(these_dists)
    print([ct_data[tuple(peaks[i])] for i in contacts])
    print([len(volumes[i]) for i in contacts])


for j, contacts in enumerate(devices):
    get_device_data(contacts)
    for i in contacts:
        renderer.sphere(
            center=peaks[i], color=UNIQUE_COLORS[j % N_COLORS], scale=2)
    input(str(j))


for k in contacts:
    renderer.sphere(center=peaks[k], color='red', scale=2)
    input(str(k))


if verbose:
    print('Using maching cubes to construct CT scaffolding')
vert, tri, _, _ = measure.marching_cubes(ct_data)
renderer = mne.viz.backends.renderer.create_3d_figure(
    size=(1200, 900), bgcolor='w', scene=False)
renderer.mesh(*vert.T, triangles=tri, color=(0.8,) * 3, opacity=0.1)
for i in range(peaks.shape[0]):
    renderer.sphere(center=peaks[i], color='blue', scale=1)
c = 'red'
renderer.sphere(center=loc, color='red', scale=1)
renderer.sphere(center=loc1, color=c, scale=5)

for peak in peaks:
    if np.sum((peak - loc)**2)**0.5 < 5:
        print(peak)

if 1:
    peak = tuple(peak)
    renderer = mne.viz.backends.renderer.create_3d_figure(
        size=(1200, 900), bgcolor='w', scene=False)
    vol = peak_to_volume(peak, ct_data, volume_max, volume_thresh)
    for loc in vol:
        v = ct_data[loc] / ct_data[peak]
        renderer.sphere(center=loc, color=(0, v, 0), scale=1)
    renderer.sphere(center=peak, color='red', scale=1)

if verbose:
    print('Using difference of guassians filter to find electrodes')
img = filters.difference_of_gaussians(ct_data, size0, spacing0)
ivert, itri, _, _ = measure.marching_cubes(
    img, level=np.quantile(img.flatten(), viz_quartile))

if verbose:
    print('Finding local maxima on the CT')
peaks = feature.peak_local_max(ct_data)
if verbose:
    print('{:.3f} of the image are local peaks'.format(
        peaks.shape[0] / ct_data.size))

locations = np.random.random((MAX_N_CONTACTS, 3)) * np.array(ct_data.shape)
booleans = np.zeros((MAX_N_CONTACTS))
# turn into an array for opimization
x0 = np.concatenate([[size0, spacing0], booleans, locations.flatten()])
if verbose:
    print(f'Optimizing {x0.size} parameters to find the best electrode '
          'locations')

def error_fun(x):
    size, spacing = np.round(np.clip(x[:2], 0, ct_shape.max())).astype(int)
    mask = x[2:MAX_N_CONTACTS + 2] >= 0
    # mask locations by whether booleans are >= 0
    locs = x[MAX_N_CONTACTS + 2:].reshape(MAX_N_CONTACTS, 3)[mask]
    locs = locs % ct_shape  # no going out of bounds
    # compute distance matrix
    dist_mat = scipy.spatial.distance_matrix(locs, locs)
    # +brightness size around loc and -brightness spacing  around loc
    value_score = 0
    for loc in locs:
        value_score += get_radius(ct_data, loc, size).mean()
        value_score -= get_radius(ct_data, loc, spacing, size).mean()
    # maximize distance so we don't get all on top of each other
    # maximize mask which is number of electrodes
    # maximize the value near the point
    # minimize the value around the point
    # maximize the spacing
    return spacing - value_score - dist_mat.mean() - mask.sum()

ct_shape = np.array(ct_data.shape)
electrode_template = make_sphere(size, spacing)
# template_shape = np.array(electrode_template.shape)

def error_fun(x):
    """Checks error at two points with the electrode template."""
    x, y, z, r, phi, theta = x
    loc0 = np.array([x, y, z])
    loc1 = np.array(polar_to_point(r, phi, theta)) + loc0
    for loc in [loc0, loc1]:
        if any([(c < 0) or (c >= bound) for c, bound in
                zip(loc, ct_shape)]):
            return np.inf
    return template_error(loc0, electrode_template, ct_data) + \
        template_error(loc1, electrode_template, ct_data)

for i in range(n_runs):
    x0 = None
    while x0 is None or np.isinf(error_fun(x0)):
        x, y, z = np.random.random(3) * ct_shape
        phi = np.random.random() * np.pi
        theta = np.random.random() * np.pi / 2
        x0 = np.array([x, y, z, spacing, phi, theta])
    res = scipy.optimize.minimize(error_fun, x0, method='Nelder-Mead',
                                  tol=1e-6)
    loc0 = np.array(res.x[:3])
    loc1 = loc0 + polar_to_point(*res.x[3:])

if electrode_templates is None:
    sizes = np.linspace(size_min, size_max, 10)
    spacings = np.linspace(spacing_min, spacing_max, 10)
    electrode_templates = [make_sphere(radius, spacing) for
                           radius, spacing in zip(sizes, spacings)]

if verbose:
    print('Matching peaks to the electrode templates')
template_errors = list()
for i, loc in enumerate(peaks.copy()[:40]):
    template_errors.append(min([template_error(loc, et, ct_data)
                                for et in electrode_templates]))
    plot_point(*loc, 10, ct_data)
    plt.gcf().suptitle(template_errors[-1])
