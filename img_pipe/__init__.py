"""Image processing software for locating intracranial electrodes."""

__version__ = '2.0'

from img_pipe.img_pipe import (check_pipeline, get_electrode_names, recon,  # noqa
                               plot_pial, label, coreg_CT_MR, mark_electrodes)  # noqa

from img_pipe.viz import (plot_brain, plot_recon_anatomy,  # noqa
                          ROI, get_rois, launch_electrode_picker)  # noqa
