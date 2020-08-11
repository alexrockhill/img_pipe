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
import numpy as np
from shutil import copyfile, move

import nibabel as nib
import scipy.io

from img_pipe.config import SUBCORTICAL_INDICES
from img_pipe.utils import (check_file, check_dir, make_dir, check_fs_vars,
                            get_fs_labels, subcort_fs2mlab)


def check_pipeline():
    """Prints the status of the pipeline.

    Prints which commands have been run and which need to be run
    """
    base_path = check_fs_vars()

    def print_status(keyword, fname):
        status = 'done ' if op.isfile(fname) else 'to do'
        spaces = ' ' * (50 - len(keyword))
        print(f'# {keyword}{spaces}# {status} #')
        print('#' * 61)

    print('#' * 61)  # fencepost
    make_dir(op.join(base_path, 'acpc'))
    print_status('Put T1 in acpc/T1.nii', op.join(base_path, 'acpc',
                                                  'T1_orig.nii'))
    print_status('Align MRI (using freeview)', op.join(base_path, 'acpc',
                                                       'T1.nii'))
    make_dir(op.join(base_path, 'CT'))
    print_status('Put CT in CT/CT.nii', op.join(base_path, 'CT', 'CT.nii'))
    make_dir(op.join(base_path, 'ieeg'))
    print_status('Put ieeg in ieeg/(name).(fif|edf|bdf|vhdr|set)',
                 op.join(base_path, 'CT', 'CT.nii'))
    print_status('img_pipe.get_electrode_names',
                 op.join(base_path, 'elecs', 'electrode_names.tsv'))
    print_status('img_pipe.recon', op.join(base_path, 'mri', 'aseg.mgz'))
    print_status('img_pipe.label', op.join(base_path, 'surf',
                                           'lh.pial.filled.mgz'))
    print_status('img_pipe.coreg_CT_MR', op.join(base_path, 'CT', 'rCT.nii'))
    print_status('img_pipe.mark_electrodes',
                 op.join(base_path, 'elecs', 'electrodes.tsv'))


def get_electrode_names(verbose=True):
    import mne
    base_path = check_fs_vars()
    fnames = [op.join(base_path, 'ieeg', fname) for fname in
              os.listdir(op.join(base_path, 'ieeg'))
              if op.splitext(fname)[-1] in
              ('.fif', '.edf', '.bdf', '.vhdr', '.set')]
    if len(fnames) > 1:
        raise ValueError('{} data files found in ieeg, expected one'.format(
            len(fnames)))
    elif len(fnames) == 0:
        if len(os.listdir(op.join(base_path, 'ieeg'))) > 0:
            for fname in os.listdir(op.join(base_path, 'ieeg')):
                raise ValueError('Extension {} not recognized, options are'
                                 'fif, edf, bdf, vhdr (brainvision) and set '
                                 '(eeglab)'.format(op.splitext(fname)[-1]))
        raise ValueError('No data files found in ieeg folder')
    fname = fnames[0]
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
    make_dir(op.join(base_path, 'elecs'))
    with open(op.join(base_path, 'elecs', 'electrode_names.tsv'), 'w') as fid:
        fid.write('name\n')
        for idx in ch_indices:
            if raw.ch_names[idx] not in ('Event', 'STI 014'):
                fid.write(f'{raw.ch_names[idx]}\n')


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
    os.system(f'mri_convert {T1_file2} {T1_mgz}')

    if verbose:
        print('Running recon all, this will take many hours')
    os.system('recon-all -s {} -sd {} -all'.format(
        os.environ['SUBJECT'], os.environ['SUBJECTS_DIR']))


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
    os.system('freeview -v {} {}:colormap=heat:opacity=0.4 '
              '-f {}:edgecolor=blue {}:edgecolor=red {}:edgecolor=blue '
              '{}:edgecolor=red {}:visible=0 {}:visible=0'.format(
                  brainmask, wm, *segs))
    # only run if precise edits were sufficient, otherwise use threshold
    if input('Run recon (Y/n)? ').lower() == 'y':
        os.system('recon-all -autorecon2-wm -autorecon3 -s {} '
                  '-no-isrunning'.format(os.environ['SUBJECT']))


def label(verbose=True):
    """Create gyri, mesh and subcortical labels.

    For meshes, the mat files made in the mesh directory can
    be imported into plotting tools like Blender or used
    directly in python or MATLAB.

    For subcortical labels, see `aseg2srf.sh`; the indices in
    `SUBCORTICAL_INDICES` correspond to the indices in
    FreeSurferColorLUT.txt` which map onto the names of the
    subcortical structures.

    Parameters
    ----------
    verbose : bool
        Whether to print text updating on the status of the function.
    """
    base_path = check_fs_vars()
    # create gyri labels directory
    gyri_labels_dir = make_dir(op.join(base_path, 'label', 'gyri'))
    # This version of mri_annotation2label uses the coarse labels
    # from the Desikan-Killiany Atlas
    for hemi in ('lh', 'rh'):
        if verbose:
            print(f'Making labels for {hemi}')
        os.system('mri_annotation2label --subject {} --hemi {} --surface pial '
                  '--outdir {}'.format(os.environ['SUBJECT'], hemi,
                                       gyri_labels_dir))

    # make meshes
    surf_dir = check_dir(op.join(base_path, 'surf'), 'recon')
    mesh_dir = make_dir(op.join(base_path, 'meshes'))
    # loop through types of mesh for plotting
    for mesh_name in ('pial', 'white', 'inflated'):
        # loop through hemispheres for this mesh, create one .mat file for each
        for hemi in ('lh', 'rh'):
            if verbose:
                print(f'Making {hemi} {mesh_name} mesh')
            mesh_surf = op.join(surf_dir, f'{hemi}.{mesh_name}')
            vert, tri = nib.freesurfer.read_geometry(mesh_surf)
            out_file = op.join(mesh_dir,
                               f'{hemi}_{mesh_name}_trivert.mat')
            out_file_struct = op.join(
                mesh_dir, '{}_{}_{}.mat'.format(os.environ['SUBJECT'],
                                                hemi, mesh_name))
            scipy.io.savemat(out_file, {'tri': tri, 'vert': vert})
            cortex = {'tri': tri + 1, 'vert': vert}
            scipy.io.savemat(out_file_struct, {'cortex': cortex})

    # make subcortical labels
    # set ascii dir name
    ascii_dir = make_dir(op.join(base_path, 'ascii'))
    # convert all ascii subcortical meshes to matlab vert, tri coords
    subcort_dir = make_dir(op.join(base_path, 'meshes', 'subcortical'))
    # get the names of the numbered regions
    number_dict = get_fs_labels()

    # tessellate all subjects freesurfer subcortical segmentations
    if verbose:
        print('::: Tesselating freesurfer subcortical segmentations '
              'from aseg using aseg2srf... :::')
    if os.system('{} -s {} -l "{}"'.format(
        op.join(op.dirname(__file__), 'aseg2srf.sh'),
        os.environ['SUBJECT'],
            ' '.join([str(idx) for idx in SUBCORTICAL_INDICES]))):
        raise RuntimeError('error in aseg2srf.sh')

    if verbose:
        print('::: Converting all ascii segmentations to matlab tri-vert :::')
    for i in SUBCORTICAL_INDICES:
        fname_srf = check_file(op.join(ascii_dir, 'aseg_{:03d}.srf'.format(i)),
                               instructions='`aseg2surf` error, please report')
        fname = fname_srf.replace('.srf', '.asc')
        move(fname_srf, fname)
        check_file(fname, instructions='Renaming aseg issue, please report')
        subcort_fs2mlab(subcort_dir, fname, number_dict[i])

    # filled pial surfaces
    for hemi in ('lh', 'rh'):
        if verbose:
            print(f'Filling pial surface for {hemi}')
        pial_fill = op.join(base_path, 'surf', f'{hemi}.pial.filled.mgz')
        if op.isfile(pial_fill):
            print(f'Filled pial surface already exists for {hemi}')
        else:
            pial_surf = op.join(base_path, 'surf', f'{hemi}.pial')
            os.system(f'mris_fill -c -r 1 {pial_surf} {pial_fill}')


def coreg_CT_MR(smooth=0., reg_type='rigid', interp='pv', xtol=0.0001,
                ftol=0.0001, overwrite=False, verbose=True):
    '''Coregistrs the anatomical MR with the CT using fsl.

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
        Whether to print text updating on the status of the function.
    '''
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


def mark_electrodes(verbose=True):
    """ Manually identify electrode locations.
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
