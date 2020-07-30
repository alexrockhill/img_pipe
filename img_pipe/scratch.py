"""Unused but useful backup functions for img_pipe."""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD (3-clause)

import os
import os.path as op
from img_pipe.utils import check_file, check_fs_vars


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
