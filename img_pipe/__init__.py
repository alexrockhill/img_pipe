"""Image processing software for locating intracranial electrodes."""

__version__ = '2.0'

from img_pipe.utils import list_rois, export_labels  # noqa
from img_pipe.img_pipe import (check_pipeline, recon, plot_pial, label,  # noqa
                               coreg_CT_MR, manual_mark_electrodes,
                               mark_electrodes, label_electrodes, warp,
                               warp_cvs)

from img_pipe.viz import ROI, get_rois, plot_brain  # noqa
