"""img_pipe intracranial eeg processing pipeline"""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#          Liberty Hamilton
#          Morgan Lee
#          David Chang
#          Anthony Fong
#          Zachary Greenberg
#          Ben Speidel
#
# License: BSD (3-clause)

import os
import os.path as op
import json
from subprocess import run
from shutil import copyfile
import numpy as np
from tqdm import tqdm

from img_pipe.config import (SUBCORTICAL_INDICES, ATLAS_DICT, VOXEL_SIZES,
                             TEMPLATES)
from img_pipe.utils import (check_file, make_dir, check_fs_vars,
                            get_ieeg_fnames, get_fs_labels, aseg_to_surf,
                            load_image_data, load_electrodes,
                            save_electrodes, get_vox_to_ras)


def check_pipeline():
    """Prints the status of the pipeline.

    Prints which commands have been run and which need to be run
    """
    base_path = check_fs_vars()

    def print_status(keyword, fname):
        if fname is None:
            status = 'to do'
        else:
            status = 'done ' if op.isfile(fname) else 'to do'
        spaces = ' ' * (50 - len(keyword))
        print(f'# {keyword}{spaces}# {status} #')
        print('#' * 61)

    print('#' * 61)  # fencepost
    make_dir(op.join(base_path, 'acpc'))
    print_status('Put T1 in acpc/T1_orig.nii', op.join(base_path, 'acpc',
                                                       'T1_orig.nii'))
    print_status('Align MRI (using freeview), save to acpc/T1.nii',
                 op.join(base_path, 'acpc', 'T1.nii'))
    make_dir(op.join(base_path, 'CT'))
    print_status('Put CT in CT/CT.nii', op.join(base_path, 'CT', 'CT.nii'))
    make_dir(op.join(base_path, 'ieeg'))
    print_status('Put ieeg in ieeg/(name).(fif|edf|bdf|vhdr|set)',
                 get_ieeg_fnames(return_first=True, verbose=False))
    print_status('img_pipe.recon', op.join(base_path, 'mri', 'aseg.mgz'))
    print_status('img_pipe.warp_to_template',
                 op.join(base_path, 'cvs', 'nlalign-aseg.mgz'))
    print_status('img_pipe.label', op.join(base_path, 'surf',
                                           'lh.pial.filled.mgz'))
    print_status('img_pipe.coreg_CT_MR', op.join(base_path, 'CT', 'rCT.nii'))
    print_status('img_pipe.mark_electrodes',
                 op.join(base_path, 'elecs', 'electrodes.tsv'))
    print_status('img_pipe.label_electrodes',
                 op.join(base_path, 'elecs', 'electrodes_labeled'))
    print_status('img_pipe.warp',
                 op.join(base_path, 'elecs', 'tmp'))
    print_status('img_pipe.export_labels', op.join(base_path, 'meshes'))


def recon(verbose=True):
    """Runs freesurfer recon-all for surface reconstruction.

    Parameters
    ----------
    verbose : bool
        Whether to print text updating on the status of the function.
    """
    base_path = check_fs_vars()

    orig_dir = make_dir(op.join(base_path, 'mri', 'orig'))

    T1_file = check_file(op.join(base_path, 'acpc', 'T1.nii'),
                         instructions='Convert T1 to nii, align to acpc '
                                      'and place at {} to fix this'.format(
        op.join(os.environ['SUBJECTS_DIR'], os.environ['SUBJECT'],
                'acpc', 'T1.nii')))
    if verbose:
        print('Copying original T1 file over to freesurfer directory')
    copyfile(T1_file, op.join(orig_dir, 'T1.nii'))

    # convert T1 to freesurfer 001.mgz format
    T1_file2 = op.join(orig_dir, 'T1.nii')
    T1_mgz = op.join(orig_dir, '001.mgz')
    if verbose:
        print('Converting to mgz format for recon-all')
    run(f'mri_convert {T1_file2} {T1_mgz}'.split(' '))

    if verbose:
        print('Running recon all, this will take many hours')
    run('recon-all -s {} -sd {} -all'.format(
        os.environ['SUBJECT'], os.environ['SUBJECTS_DIR']).split(' '))


def warp_to_template(template='cvs_avg35_inMNI152', n_jobs=1, verbose=True):
    """Warps electrodes to a common atlas.

    Parameters
    ----------
    template : str, optional
        Which atlas brain to use. Must be one of ['V1_average',
        'cvs_avg35', 'cvs_avg35_inMNI152', 'fsaverage',
        'fsaverage3', 'fsaverage4', 'fsaverage5', 'fsaverage6',
        'fsaverage_sym']
    picks: list
        An optional list of electrodes to name if you are using different
        atlases for different groups.
    n_jobs: int
        The number of cores to use for parallel computing to speed up the
        process.
    verbose : bool
        Whether to print text updating on the status of the function.

    """
    if template not in TEMPLATES:
        raise ValueError(f'Template must be in {TEMPLATES}, got {template}')
    if verbose:
        print(f'Using {template} as the template for electrode warping')

    if verbose:
        print('Computing combined volumetric surface (cvs) registration '
              'this will take many hours...')
    run(['mri_cvs_register', '--mov', os.environ['SUBJECT'],
         '--templatedir', op.join(os.environ['FREESURFER_HOME'],
                                  'subjects'),
         '--template', template, '--nocleanup', '--openmp', str(n_jobs)])


def plot_pial(verbose=True):
    """This edits the pial surface according to the freesurfer tutorial.

    https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/WhiteMatterEdits_freeview
    Follow instructions to edit pial surface and then the recon will be rerun.

    Note: if the recon totally failed as it might for an epilepsy patient
    who has a lot of brain matter missing, use Tools>>Threshold Volume...
    and set the value in the low box to apply to the white matter brainmask
    or both in order to get a more accurate recon of those quantities.

    Parameters
    ----------
    verbose : bool
        Whether to print text updating on the status of the function.
    """
    base_path = check_fs_vars()
    # get file paths to check and edit
    brainmask = check_file(op.join(base_path, 'mri', 'brainmask.mgz'), 'recon')
    wm = check_file(op.join(base_path, 'mri', 'wm.mgz'), 'recon')
    segs = list()
    for seg in ('lh.white', 'lh.pial', 'rh.white', 'rh.pial',
                'rh.inflated', 'lh.inflated'):
        segs.append(check_file(op.join(base_path, 'surf', seg), 'recon'))
    run('freeview -v {} {}:colormap=heat:opacity=0.4 '
        '-f {}:edgecolor=blue {}:edgecolor=red {}:edgecolor=blue '
        '{}:edgecolor=red {}:visible=0 {}:visible=0'.format(
            brainmask, wm, *segs).split(' '))
    # only run if precise edits were sufficient, otherwise use threshold
    if input('Run recon (Y/n)? ').lower() == 'y':
        run('recon-all -autorecon2-wm -autorecon3 -s {} '
            '-no-isrunning'.format(os.environ['SUBJECT']).split(' '))


def label(atlas=None, sigma=1, overwrite=False, verbose=True):
    """Create gyri, subcortical and pial labels.

    See `aseg2srf.sh`; the indices in `SUBCORTICAL_INDICES`
    correspond to the indices in FreeSurferColorLUT.txt` which
    map onto the names of the subcortical structures.

    Parameters
    ----------
    atlas: str
        The segementation atlas to use; 'desikan-killiany', 'DKT' or
        'destrieux'. If none is specified, all are computed. Specifying only
        one can save time.
    sigma: float
        The amount of smoothing of the volumes created from the segmentation.
    overwrite: bool
        Whether to overwrite the target filepath
    verbose: bool
        Whether to print text updating on the status of the function
    """
    base_path = check_fs_vars()
    if op.isfile(op.join(base_path, 'surf', 'lh.pial.filled.mgz')) and \
            not overwrite:
        raise ValueError('img_pipe.label has already been run and '
                         'overwrite=False')

    # get the names of the numbered regions
    if atlas is None:
        atlas_dict = ATLAS_DICT
    else:
        if atlas not in ATLAS_DICT:
            raise ValueError(f'Atlas {atlas} not recognized, options are '
                             '`{}`'.format('`, `'.join(ATLAS_DICT.keys())))
        atlas_dict = {atlas: ATLAS_DICT[atlas]}
    number_dict = get_fs_labels()
    vox_to_ras, _ = get_vox_to_ras()
    # make gyri labels
    for atlas, seg in atlas_dict.items():
        aseg = load_image_data('mri', f'{seg}+aseg.mgz', 'img_pipe.recon')
        roi_idxs = np.unique(aseg[aseg > 0]).astype(int)  # > 0 == no unknown
        assert all([idx in roi_idxs for idx in SUBCORTICAL_INDICES])
        if verbose:
            print(f'Making {atlas} volumes from segmentation for '
                  '{}'.format(', '.join([number_dict[idx]
                                         for idx in roi_idxs])))
        label_dir = make_dir(op.join(base_path, 'label', atlas))
        for idx in tqdm(roi_idxs):
            out_fname = op.join(label_dir, number_dict[idx])
            if op.isfile(out_fname) and not overwrite:
                print(f'{out_fname} already exists, skipping')
            else:
                aseg_to_surf(out_fname, aseg, idx, trans=vox_to_ras,
                             sigma=sigma, verbose=verbose)
    # make filled pial surfaces
    for hemi in ('lh', 'rh'):
        if verbose:
            print(f'Filling pial surface for {hemi}')
        pial_fill = op.join(base_path, 'surf', f'{hemi}.pial.filled.mgz')
        if op.isfile(pial_fill) and not overwrite:
            print(f'Filled pial surface already exists for {hemi}')
        else:
            pial_surf = op.join(base_path, 'surf', f'{hemi}.pial')
            run(f'mris_fill -c -r 1 {pial_surf} {pial_fill}'.split(' '))


def coreg_CT_MR(smooth=0., reg_type='rigid', interp='pv', xtol=0.0001,
                ftol=0.0001, overwrite=False, verbose=True):
    """Coregistrs the anatomical MR with the CT using nibabel.

    Parameters are arguments for
    nipy.algorithms.registration.histogram_registration.HistogramRegistration
    (for more information, see help for
    nipy.algorithms.registration.optimizer).

    Parameters
    ----------
    smooth : float
        a smoothing parameter in mm
    reg_type : {'rigid', 'affine'}
        Registration type
    interp : {'pv', 'tri'}
        changes the interpolation method for resampling
    xtol : float
        tolerance parameter for function minimization
    ftol : float
        tolerance parameter for function minimization
    overwrite : bool
        Whether to overwrite the target filepath
    verbose : bool
        Whether to print text updating on the status of the function
    """
    from nipy import load_image, save_image
    from nipy.algorithms.registration.histogram_registration import (
        HistogramRegistration)
    from nipy.core.api import AffineTransform
    from nipy.algorithms.resample import resample

    base_path = check_fs_vars()

    in_fname = check_file(op.join(base_path, 'CT', 'CT.nii'),
                          instructions='Convert CT to nii using `mri_convert` '
                                       'and copy to {}'.format(
        op.join(os.environ['SUBJECTS_DIR'], os.environ['SUBJECT'],
                'CT', 'CT.nii')))
    ref_fname = check_file(op.join(base_path, 'mri', 'orig.mgz'), 'recon')
    out_fname = op.join(base_path, 'CT', 'rCT.nii')

    if op.isfile(out_fname) and not overwrite:
        raise ValueError(f'{out_fname} exists, use `overwrite=True` to '
                         'overwrite')

    if verbose:
        print(f'Computing registration from {in_fname} to {ref_fname}'
              '\n\nIf this is taking too long/doesn\'t work you may want '
              'you may want to translate the CT to be over the MR in freeview '
              'using `cd $SUBJECTS_DIR; freeview {$SUBJECT}/mri/orig/T1.nii '
              '{$SUBJECT}/CT/CT.nii` adjusting the opacity and using tools>'
              'Transform Volume and translate and scale to put the CT '
              'roughly in the right place before saving over the original CT')

    ct_img = load_image(in_fname)
    mri_img = load_image(ref_fname)

    # Compute registration
    ct_to_mri_reg = HistogramRegistration(ct_img, mri_img, similarity='nmi',
                                          smooth=smooth, interp=interp)
    aff = ct_to_mri_reg.optimize(reg_type).as_affine()

    ct_to_mri = AffineTransform(ct_img.coordmap.function_range,
                                mri_img.coordmap.function_range, aff)
    reg_CT = resample(ct_img, mri_img.coordmap,
                      ct_to_mri.inverse(), mri_img.shape)

    if verbose:
        print(f'Saving registered CT image as {out_fname}')
    save_image(reg_CT, out_fname)


def manual_mark_electrodes(verbose=True):
    """Manually identify electrode locations.

    Parameters
    ----------
    verbose : bool
        Whether to print text updating on the status of the function.

    """
    from img_pipe.viz import launch_electrode_picker
    base_path = check_fs_vars()
    make_dir(op.join(base_path, 'elecs'))
    if verbose:
        print('Launching electrode picker')
    launch_electrode_picker()


def mark_electrodes(verbose=True):
    """Automatically identify electrode positions and assign with GUI

    The default arguments were chosen to work for 10-20 stereoeeg
    electrodes with 4-16 contacts (~3 mm diameter) and ~5 mm spacing
    or one or two ECoG grids with ~40 contacts.

    Parameters
    ----------
    verbose: bool
        Whether to print text updating on the status of the function.

    """
    from img_pipe.viz import launch_electrode_gui
    base_path = check_fs_vars()
    make_dir(op.join(base_path, 'elecs'))
    if verbose:
        print('Launching electrode graphical user interface')
    launch_electrode_gui()


def label_electrodes(atlas='desikan-killiany', picks=None,
                     overwrite=False, verbose=True):
    """Automatically labels electrodes based on the freesurfer recon.

    The atlases are described here:
    https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation
    The allowed atlases are 'desikan-killiany' and 'destrieux'

    Parameters
    ----------
    atlas : str
        The atlas to use for labeling of electrodes.
    picks: list
        An optional list of electrodes to name if you are using different
        atlases for different groups.
    overwrite : bool
        Whether to overwrite the target filepath
    verbose : bool
        Whether to print text updating on the status of the function.

    """
    if atlas not in ATLAS_DICT:
        raise ValueError('Atlas must be in {}, got {}'.format(
            list(ATLAS_DICT.keys()), atlas))
    elec_matrix = load_electrodes(verbose=verbose)
    fs_labels = get_fs_labels()
    img_data = load_image_data('mri', ATLAS_DICT[atlas] + '+aseg.mgz',
                               verbose=verbose)
    if verbose:
        print('Labeling electrodes...')
    for name in elec_matrix:
        if picks is None or name in picks:
            r, a, s, _, label = elec_matrix[name]
            vx, vy, vz = np.round([r, a, s]).astype(int) + VOXEL_SIZES // 2
            this_label = fs_labels[img_data[vx, vy, vz].astype(int)]
            label_dict = dict() if label == 'n/a' else json.loads(label)
            if atlas in label_dict and not overwrite:
                raise ValueError(f'{atlas} label already assigned for {name}'
                                 f'overwrite=False')
            label_dict[atlas] = this_label
            elec_matrix[name][4] = json.dumps(label_dict)
    save_electrodes(elec_matrix, atlas=atlas, verbose=verbose)


def warp(template='cvs_avg35_inMNI152', n_jobs=1, overwrite=False,
         verbose=True):
    """Warps electrodes to a common atlas.

    Parameters
    ----------
    template : str, optional
        Which atlas brain to use. Must be one of ['V1_average',
        'cvs_avg35', 'cvs_avg35_inMNI152', 'fsaverage',
        'fsaverage3', 'fsaverage4', 'fsaverage5', 'fsaverage6',
        'fsaverage_sym']
    n_jobs: int
        The number of cores to use for parallel computing to speed up the
        process.
    overwrite : bool
        Whether to overwrite the target filepath
    verbose : bool
        Whether to print text updating on the status of the function.

    """
    if template not in TEMPLATES:
        raise ValueError(f'Template must be in {TEMPLATES}, got {template}')
    if verbose:
        print(f'Using {template} as the template for electrode warping')

    base_path = check_fs_vars()
    warped_fname = op.join(base_path, 'elecs', f'electrodes_{template}.tsv')
    if op.isfile(warped_fname) and not overwrite:
        raise ValueError(f'Electrodes are already warped to {template} and '
                         'overwrite is False')
    check_file(
        op.join(base_path, 'cvs', f'final_CVSmorphed_to{template}_aseg.mgz'),
        instructions='Run `mri_cvs_register` to create it')
    morph_fname = check_file(
        op.join(base_path, 'cvs',
                f'combined_to{template}_elreg_afteraseg-norm.tm3d'),
        instructions='Check `mri_cvs_register` for errors and '
        'make sure to use the flag --nocleanup')
    template_fname = check_file(
        op.join(os.environ['FREESURFER_HOME'], 'subjects', template,
                'mri', 'brain.mgz'), instructions='Check freesurfer install')
    elec_matrix = load_electrodes(verbose=verbose)
    tmp_fname = op.join(base_path, 'cvs', 'tmp.txt')
    with open(tmp_fname, 'w') as fid:
        for name in elec_matrix:
            r, a, s = elec_matrix[name][:3]
            vx, vy, vz = np.array([r, a, s]) + VOXEL_SIZES // 2
            fid.write(f'{vx}\t{vy}\t{vz}\n')
    out_fname = op.join(base_path, 'cvs', 'out.txt')
    run(f'applyMorph --template {template_fname} '
        f'--transform {morph_fname} '
        f'tract_point_list {tmp_fname} {out_fname} nearest', shell=True)
    warped_elec_matrix = dict()
    with open(out_fname, 'r') as fid:
        for name in elec_matrix:
            vx, vy, vz = [float(p) for p in
                          fid.readline().rstrip().split()]
            r, a, s = np.array([vx, vy, vz]) - VOXEL_SIZES // 2
            warped_elec_matrix[name] = [r, a, s] + elec_matrix[name][3:]
    save_electrodes(warped_elec_matrix, template=template)
    os.remove(tmp_fname)
    os.remove(out_fname)
