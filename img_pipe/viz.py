''' This module contains a function (ctmr_brain_plot) that takes as
 input a 3d coordinate set of triangular mesh vertices (vert) and
 an ordered list of their indices (tri), to produce a 3d surface
 model of an individual brain. Assigning the result of the plot
 to a variable enables interactive changes to be made to the OpenGl
 mesh object. Default shading is phong point shader (shiny surface).

 usage: from ctmr_brain_plot import *
        dat = scipy.io.loadmat('/path/to/lh_pial_trivert.mat');
        mesh = ctmr_brain_plot(dat['tri'], dat['vert']);
        mlab.show()

 A second function contained in this module can be used to plot electrodes
 as glyphs (spehres) or 2d circles. The function (el_add) takes as input
 a list of 3d coordinates in freesurfer surface RAS space and plots them
 according to the color and size parameters that you provide.

 usage: elecs = scipy.io.loadmat('/path/to/hd_grid.mat')['elecmatrix'];
        points = el_add(elecs, color = (1, 0, 0), msize = 2.5);
        mlab.show()

Modified for use in python from MATLAB code originally written by
Kai Miller and Dora Hermes (ctmr_gui,
see https://github.com/dorahermes/Paper_Hermes_2010_JNeuroMeth)

'''

import os.path as op
import scipy.io

import numpy as np

import scipy
from scipy.ndimage import binary_closing

from img_pipe.config import (VOXEL_SIZES, CT_MIN_VAL, MAX_N_GROUPS,
                             CMAP, SUBCORTICAL_INDICES, ZOOM_STEP_SIZE,
                             ELEC_PLOT_SIZE, CORTICAL_SURFACES)
from img_pipe.utils import (check_fs_vars, check_file, check_hemi, get_surf,
                            get_azimuth, get_fs_labels, get_fs_colors,
                            load_electrode_names, load_electrodes,
                            save_electrodes, load_image_data)

import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt  # noqa
from matplotlib import cm  # noqa
import matplotlib.colors as mcolors  # noqa
from matplotlib.widgets import Slider, Button  # noqa

from PyQt5 import QtCore, QtGui, Qt  # noqa
from PyQt5.QtCore import pyqtSlot  # noqa
from PyQt5.QtWidgets import (QApplication, QMainWindow,  # noqa
                             QVBoxLayout, QHBoxLayout, QLabel,  # noqa
                             QInputDialog, QMessageBox, QWidget,  # noqa
                             QListView, QSlider, QPushButton,
                             QComboBox, QPlainTextEdit)  # noqa
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # noqa

import mayavi  # noqa
from mayavi import mlab  # noqa

ELECTRODE_COLORS = mcolors.LinearSegmentedColormap.from_list(
    'elec_colors', np.vstack(
        (cm.Set1(np.linspace(0., 1, MAX_N_GROUPS // 2 + 1 - MAX_N_GROUPS % 2)),
         cm.Set2(np.linspace(0., 1, MAX_N_GROUPS // 2)))))
NORM = mpl.colors.Normalize(vmin=0, vmax=MAX_N_GROUPS)


def _ctmr_gauss_plot(tri, vert, color=(0.8, 0.8, 0.8), elecs=None,
                     weights=None, opacity=1.0, representation='surface',
                     line_width=1.0, gsp=10, cmap=mpl.cm.get_cmap('RdBu_r'),
                     show_colorbar=True, new_fig=True, vmin=None, vmax=None,
                     ambient=0.4225, specular=0.333, specular_power=66,
                     diffuse=0.6995, interpolation='phong'):
    """This function plots the 3D brain surface mesh

    Parameters
    ----------
        color : tuple
            (n,n,n) tuple of floats between 0.0 and 1.0,
            background color of brain
        elecs : array-like
            [nchans x 3] matrix of electrode coordinate values in 3D
        weights : array-like
            [nchans x 1] - if [elecs] is also given,
            this will color the brain vertices according to these weights
        msize : float
            size of the electrode.  default = 2
        opacity : float (0.0 - 1.0)
            opacity of the brain surface (value from 0.0 - 1.0)
        cmap : str or mpl.colors.LinearSegmentedColormap
            colormap to use when plotting gaussian weights with [elecs]
            and [weights]
        representation : {'surface', 'wireframe'}
            surface representation
        line_width : float
            width of lines for triangular mesh
        gsp : float
            gaussian smoothing parameter, larger makes electrode activity
            more spread out across the surface if specified

    Returns
    -------
    mesh : mayavi mesh (actor)
    mlab : mayavi mlab scene
    """
    # if color is another iterable, make it a tuple.
    color = tuple(color)

    brain_color = []

    if elecs is not None:
        brain_color = np.zeros(vert.shape[0],)
        for i in np.arange(elecs.shape[0]):
            b_z = np.abs(vert[:, 2] - elecs[i, 2])
            b_y = np.abs(vert[:, 1] - elecs[i, 1])
            b_x = np.abs(vert[:, 0] - elecs[i, 0])
            gauss_wt = np.nan_to_num(
                weights[i] * np.exp((-(b_x ** 2 + b_z ** 2 + b_y ** 2)) / gsp))
            brain_color = brain_color + gauss_wt

        # scale the colors so that it matches the weights that were passed in
        brain_color *= (np.abs(weights).max() / np.abs(brain_color).max())
        if vmin is None and vmax is None:
            vmin, vmax = -np.abs(brain_color).max(), np.abs(brain_color).max()

    # plot cortex and begin display
    if new_fig:
        mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1), size=(1200, 900))

    if elecs is not None:
        kwargs = {}
        if type(cmap) == str:
            kwargs.update(colormap=cmap)

        mesh = mlab.triangular_mesh(vert[:, 0], vert[:, 1], vert[:, 2], tri,
                                    representation=representation,
                                    opacity=opacity, line_width=line_width,
                                    scalars=brain_color, vmin=vmin, vmax=vmax,
                                    **kwargs)

        if type(cmap) == mpl.colors.LinearSegmentedColormap:
            mesh.module_manager.scalar_lut_manager.lut.table = \
                (cmap(np.linspace(0, 1, 255)) * 255).astype('int')
    else:
        mesh = mlab.triangular_mesh(vert[:, 0], vert[:, 1], vert[:, 2], tri,
                                    color=color, representation=representation,
                                    opacity=opacity, line_width=line_width)

    # cell_data = mesh.mlab_source.dataset.cell_data
    # cell_data.scalars = brain_color
    # cell_data.scalars.name = 'Cell data'
    # cell_data.update()

    # mesh2 = mlab.pipeline.set_active_attribute(mesh,
    # cell_scalars='Cell data')
    # mlab.pipeline.surface(mesh)
    if weights is not None and show_colorbar:
        mlab.colorbar()

    # change OpenGL mesh properties for phong point light shading
    mesh.actor.property.ambient = ambient
    mesh.actor.property.specular = specular
    mesh.actor.property.specular_power = specular_power
    mesh.actor.property.diffuse = diffuse
    mesh.actor.property.interpolation = interpolation
    # mesh.scene.light_manager.light_mode = 'vtk'
    if opacity < 1.0:
        mesh.scene.renderer.set(use_depth_peeling=True)
        # maximum_number_of_peels=100, occlusion_ratio=0.0

    # Make the mesh look smoother
    for child in mlab.get_engine().scenes[0].children:
        poly_data_normals = child.children[0]
        # Feature angle says which angles are considered hard corners
        poly_data_normals.filter.feature_angle = 80.0

    return mesh, mlab


def add_electrodes(elecs, color=(1., 0., 0.), msize=2, labels=None,
                   label_offset=-1.0, ambient=0.3261, specular=1,
                   specular_power=16, diffuse=0.6995, interpolation='phong',
                   **kwargs):
    '''This function adds the electrode matrix [elecs] (nchans x 3) to
    the scene.

    Parameters
    ----------
        elecs : array-like
            [nchans x 3] matrix of electrode coordinate values in 3D
        color : tuple (triplet) or numpy array
            Electrode color is either a triplet (r, g, b),
            or a numpy array with the same shape as [elecs]
            to plot one color per electrode
        msize : float
            size of the electrode.  default = 2
        label_offset : float
            how much to move the number labels out by (so not blocked
            by electrodes)
        **kwargs :
            any other keyword arguments that can be passed to points3d
    '''

    # Get the current keyword arguments
    cur_kwargs = dict(color=color, scale_factor=msize, resolution=25)

    # Allow the user to override the default keyword arguments using kwargs
    cur_kwargs.update(kwargs)

    # plot the electrodes as spheres
    # If we have one color for each electrode, color them separately
    if type(color) is np.ndarray:
        if color.shape[0] == elecs.shape[0]:
            # for e in np.arange(elecs.shape[0]):
            #     points = mlab.points3d(elecs[e,0], elecs[e,1], elecs[e,2],
            #                            scale_factor=msize,
            #                            color=tuple(color[e,:]),
            #                            resolution=25)
            unique_colors = np.array(list(set([tuple(row) for row in color])))
            for individual_color in unique_colors:
                indices = np.where((color == individual_color).all(axis=1))[0]
                cur_kwargs.update(color=tuple(individual_color))
                points = mlab.points3d(elecs[indices, 0], elecs[indices, 1],
                                       elecs[indices, 2], **cur_kwargs)
        else:
            print('Warning: color array does not match the size of '
                  'the electrode matrix')

    # Otherwise, use the same color for all electrodes
    else:
        points = mlab.points3d(elecs[:, 0], elecs[:, 1], elecs[:, 2],
                               **cur_kwargs)

    # Set display properties
    points.actor.property.ambient = ambient
    points.actor.property.specular = specular
    points.actor.property.specular_power = specular_power
    points.actor.property.diffuse = diffuse
    points.actor.property.interpolation = interpolation
    # points.scene.light_manager.light_mode = 'vtk'

    if labels is not None:
        for label_idx, label in enumerate(labels):
            mayavi.mlab.text3d(elecs[label_idx, 0] + label_offset,
                               elecs[label_idx, 1],
                               elecs[label_idx, 2],
                               str(label), orient_to_camera=True)
    return points, mlab


def get_elecs_anat(region):
    base_path = check_fs_vars()
    tdt_fname = check_file(op.join(base_path, 'elecs', 'TDT_elecs_all.mat'))
    tdt = scipy.io.loadmat(tdt_fname)
    return tdt['elecmatrix'][np.where(tdt['anatomy'][:, 3] == region)[0], :]


def _ctmr_plot(hemi, elecs, weights=None, interactive=False):
    base_path = check_fs_vars()
    hemi_data_fname = check_file(op.join(base_path, 'meshes',
                                         f'{hemi}_pial_trivert.mat'))
    hemi_array = scipy.io.loadmat(hemi_data_fname)
    if weights is None:
        weights = np.ones((elecs.shape[0])) * -1.
    mesh, mlab = _ctmr_gauss_plot(hemi_array['tri'], hemi_array['vert'],
                                  elecs=elecs, weights=weights,
                                  color=(0.8, 0.8, 0.8), cmap='RdBu')

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


class ROI:
    def __init__(self, name, color=None, opacity=1.0,
                 representation='surface', gaussian=False, template=None):
        """Class defining a region of interest (ROI) mesh.

        This could be, for example, a mesh for the left hippocampus,
        or a mesh for the right superior temporal gyrus.

        Parameters
        ----------
        name : str
            The name of the ROI
            Either an ROI in `FreeSurferColorLUT.txt` or 'Left-'/'Right-' +
            'Pial'/ 'Inflated'/'White'.
        color : tuple
            Tuple for the ROI's color where each value is between 0.0 and 1.0.
        opacity : float (between 0.0 and 1.0)
            opacity of the mesh, between 0.0 and 1.0
        representation : {'surface', 'wireframe'}
            surface representation type
        gaussian: bool
            specifies how to represent electrodes and their weights on
            this mesh, note that setting gaussian to True means the mesh
            color cannot be specified by the user and will instead
            use colors from the loaded colormap.
        template : None or str
            Name of the template to use if plotting electrodes on an
            atlas brain. e.g. 'cvs_avg35_inMNI152'

        Attributes
        ----------
        name : str
        color : tuple
        opacity : float (between 0.0 and 1.0)
        representation : {'surface', 'wireframe'}
        gaussian : bool
        """
        base_path = check_fs_vars()
        number_dict = get_fs_labels()
        color_dict = get_fs_colors()

        if name not in number_dict and name not in CORTICAL_SURFACES:
            raise ValueError(f'Name {name} not recongized')
        # change number label to name
        if isinstance(name, int):
            name = number_dict[name]
        self.name = name
        if self.name in CORTICAL_SURFACES:
            hemi, roi = name.split('-')
            if hemi == 'Left':
                self.hemi = 'lh'
            elif hemi == 'Right':
                self.hemi = 'rh'
            self.mesh = get_surf(self.hemi, roi.lower(), template=template)
        else:
            if 'Left' in name:
                self.hemi = 'lh'
            elif 'Right' in name:
                self.hemi = 'rh'
            else:
                self.hemi = 'both'
            # check if subcortical
            if self.name in [number_dict[idx] for idx in SUBCORTICAL_INDICES]:
                fname = op.join(base_path, 'meshes', 'subcortical',
                                f'{self.name}_subcort_trivert.mat')
            else:
                fname = op.join(base_path, 'meshes',
                                f'{self.name}_trivert.mat')
            self.mesh = scipy.io.loadmat(fname)
        if color is None:
            if name in color_dict:
                self.color = (c / 255 for c in color_dict[name])
            else:
                self.color = (0.8, 0.8, 0.8)
        self.opacity = opacity
        self.representation = representation
        self.gaussian = gaussian

    def get_kwargs(self):
        return dict(color=self.color, opacity=self.opacity,
                    representation=self.representation)


def get_rois(group='all', opacity=1.0, representation='surface'):
    """ Get the subcortial regions of interest

    Parameters
    ----------
    group : str
        'all' for pial and subcortial structures
        'pial' for just pial surfaces
        'subcortical' for just subcortical structures
        'inflated' for left and right inflated
        'white' for left and right white matter
    opacity : float (between 0.0 and 1.0)
        opacity of the mesh, between 0.0 and 1.0
    representation : {'surface', 'wireframe'}
        surface representation type
    """
    if group in ('all', 'pial', 'inflated', 'white'):
        name = 'Pial' if group == 'all' else group.capitalize()
        cortex = [ROI(f'{hemi}-{name}', opacity=opacity,
                      representation=representation)
                  for hemi in ('Left', 'Right')]
    if group in ('all', 'subcortical'):
        subcortical = [ROI(idx, opacity=opacity, representation=representation)
                       for idx in SUBCORTICAL_INDICES]
    if group == 'all':
        return cortex + subcortical
    elif group == 'subcortical':
        return subcortical
    elif group in ('pial', 'inflated', 'white'):
        return cortex
    else:
        raise ValueError(f'Unrecognized group {group}')


def plot_brain(rois=None, elecs=None, weights=None, cmap='RdBu',
               show_fig=True, screenshot=False, helper_call=False,
               vmin=None, vmax=None, azimuth=None, elevation=90,
               opacity=1.0):
    """Plots multiple meshes on one figure.
    Defaults to plotting both hemispheres of the pial surface.

    Parameters
    ----------
    rois : list
        List of roi objects created like so:
        hipp_roi = ROI(name='Left-Hippocampus', color=(0.5,0.1,0.8)))
        See `FreeSurferColorLUT.txt` for available ROI names.
        Defaults to [ROI('Left-Pial'), ROI('Right-Pial')]
    elecs : array-like
        [nchans x 3] electrode coordinate matrix
    weights : array-like
        [nchans x 3] weight matrix associated with the electrode
        coordinate matrix
    cmap : str
        String specifying what colormap to use
    show_fig : bool
        whether to show the figure in an interactive window
    screenshot: bool
        whether to save a screenshot and show using matplotlib
        (usually inline a notebook)
    helper_call : bool
        if plot_brain being used as a helper subcall,
        don't close the mlab instance
    vmin : float
        Minimum color value when using cmap
    vmax : float
        Maximum color value when using cmap
    azimuth : float
        Azimuth for brain view.  By default (if azimuth=None),
        this will be set automatically to the left side for hemi='lh',
        right side for hemi='rh', or front view for 'pial'
    elevation : float
        Elevation for brain view. Default: 90
    opacity : float
        Opaqueness between 0 and 1.
    Returns
    -------
    mlab : mayavi mlab scene

    Example
    -------
    >>> from img_pipe import roi, get_elecs, plot_brain
    >>> pial = ROI('pial', (0.6, 0.3, 0.6), 0.1, 'wireframe', True),
    >>> hipp = ROI('Left-Hippocampus', (0.5, 0.1, 0.8), 1.0, 'surface', True)
    >>> elecs = get_elecs()['elec_matrix']
    >>> plot_brain(rois=[pial, hipp], elecs=elecs,
                   weights=np.random.uniform(0, 1, (elecs.shape[0])))
    """

    if rois is None:
        rois = get_rois('pial')
    from mayavi import mlab

    mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1), size=(1200, 900))

    # if there are any rois with gaussian set to True,
    # don't plot any elecs as points3d, to avoid mixing gaussian representation
    # if needed, simply add the elecs by calling add_electrodes
    for roi in rois:
        # setup kwargs for ctmr_brain_plot.ctmr_gauss_plot()
        kwargs = roi.get_kwargs()
        kwargs.update(color=roi.color, new_fig=False, cmap=cmap)
        if roi.gaussian:
            kwargs.update(elecs=elecs, weights=weights, vmin=vmin, vmax=vmax)

        # default roi_name of 'pial' plots both hemispheres' pial surfaces
        _ctmr_gauss_plot(roi.mesh['tri'], roi.mesh['vert'],
                         opacity=opacity, new_fig=False)

    if any([roi.gaussian for roi in rois]) and elecs is not None:
        # if elecmatrix passed in but no weights specified,
        # default to all ones for the electrode color weights
        if weights is None:
            add_electrodes(elecs)
        else:
            # Map the weights onto the current colormap
            elec_colors = cm.get_cmap(cmap)(weights)[:, :3]
            add_electrodes(elecs, color=elec_colors)

    if azimuth is None:
        azimuth = get_azimuth(rois)

    mlab.view(azimuth, elevation)

    if screenshot:
        arr = mlab.screenshot(antialiased=True)
        plt.figure(figsize=(20, 10))
        plt.imshow(arr, aspect='equal')
        plt.axis('off')
        plt.show()
    else:
        arr = []

    if show_fig:
        mlab.show()

    if not helper_call and not show_fig:
        mlab.close()

    return mlab


def plot_recon_anatomy(hemi='both', verbose=True):
    """Plot the anatomy of the reconstruction using mayavi."""
    base_path = check_fs_vars()
    for hemi in check_hemi(hemi):
        vert_data_fname = check_file(op.join(base_path, 'meshes',
                                             f'{hemi}_pial_trivert.mat'))
        vert_data = scipy.io.loadmat(vert_data_fname)
        tdt_fname = check_file(op.join(base_path, 'elecs',
                                       'TDT_elecs_all.mat'))
        tdt = scipy.io.loadmat(tdt_fname)

        # Plot the pial surface
        mesh, mlab = _ctmr_gauss_plot(vert_data['tri'], vert_data['vert'],
                                      color=(0.8, 0.8, 0.8))

        # Make a list of electrode numbers
        elec_numbers = np.arange(tdt['elec_matrix'].shape[0]) + 1

        # Find all the unique brain areas in this subject
        brain_areas = np.unique(tdt['anatomy'][:, 3])

        # plot the electrodes in each brain area
        for b_area in brain_areas:
            # Add relevant extra information to the label if needed
            label = b_area[0]
            if not any([label.startswith(name) for name in
                        ('ctx', 'Left', 'Right', 'Brain', 'Unknown')]):
                label = f'ctx-{hemi}-{label}'
                if verbose:
                    print(label)
            if label:
                electrode_color = np.array(CMAP[label]) / 255.
                add_electrodes(
                    np.atleast_2d(
                        tdt['elecmatrix'][tdt['anatomy'][:, 3] == label, :]),
                    color=tuple(electrode_color),
                    numbers=elec_numbers[tdt['anatomy'][:, 3] == label])
    mlab.show()
    return mesh, mlab


class ComboBox(QComboBox):
    """Fixes on changed bug.

    When a custom group is chosen, this is linked to the first item (auto)
    color, this puts it's color back to the auto suggestion
    """
    clicked = QtCore.pyqtSignal()

    def showPopup(self):
        self.clicked.emit()
        super(ComboBox, self).showPopup()


class SlicePlots(FigureCanvasQTAgg):
    """Initializes figure in pyqt for slice plots."""

    def __init__(self, parent=None, width=16, height=10, dpi=300):
        self.fig, self.axes = plt.subplots(2, 3, figsize=(width, height),
                                           dpi=dpi)
        super(SlicePlots, self).__init__(self.fig)


class ElectrodePicker(QMainWindow):
    """Pick electrodes manually using a coregistered MRI and CT."""

    def __init__(self, *args, **kwargs):
        """ Initialize the electrode picker.

        Images will be displayed using orientation information
        obtained from the image header. Images will be resampled to dimensions
        [256,256,256] for display.
        We will also listen for keyboard and mouse events so the user can
        interact with each of the subplot panels (zoom/pan) and
        add/remove electrodes with a keystroke.

        Parameters
        ----------
        verbose : bool
            Whether to print text updating on the status of the function.

        Attributes
        ----------
        img_data : np.array
            Data from brain.mgz T1 MRI scan
        ct_data : np.array
            Data from rCT.nii registered CT scan
        pial_data : np.array
            Filled pial image
        elec_data : np.array
            Mask for the electrodes
        bin_mat : array-like
            Temporary mask for populating elec_data
        device_idx : int
            Index of current device that has been added
        device_name : str
            Name of current device
        devices : list
            List of devices (grids, strips, depths)
        elec_idx_list : dict
            Indexed by device_name, which number electrode we are on for
            that particular device
        elec_matrix : dict
            Dictionary of electrode coordinates
        elec_added : bool
            Whether we're in an electrode added state
        current_slice : array-like
            Which 3D slice coordinate the user clicked
        fig : figure window
            The current figure window
        im : np.array
            Contains data for each axis with MRI data values.
        cursor : array-like
            Cross hair
        cursor2 : array-like
            Cross hair
        ax : matplotlib.pyplot.axis
            which of the axes we're on
        contour : list of bool
            Whether pial surface contour is displayed in each view
        pial_surf_on : bool
            Whether pial surface is visible or not
        T1_on : bool
            Whether T1 is visible or not
        ct_slice : {'s','c','a'}
            How to slice CT maximum intensity projection
            (sagittal, coronal, or axial)
        """
        # initialize QMainWindow class
        super(ElectrodePicker, self).__init__(*args, **kwargs)

        # load imaging data
        self.base_path = check_fs_vars()
        self.load_image_data()

        # initialize electrode data
        self.elec_index = 0
        self.elec_names = load_electrode_names()

        self.elec_radius = int(np.mean(ELEC_PLOT_SIZE) // 100)
        # add already marked electrodes if they exist
        self.elec_matrix = load_electrodes()
        for name in self.elec_matrix:
            if name not in self.elec_names:
                self.elec_names.append(name)

        self.pial_surf_on = True  # Whether pial surface is visible or not
        self.T1_on = True  # Whether T1 is visible or not

        # GUI design
        self.setWindowTitle('Electrode Picker')

        self.make_slice_plots()

        button_hbox = self.get_button_bar()
        slider_hbox = self.get_slider_bar()

        self.elec_list = QListView()
        self.elec_list.setSelectionMode(Qt.QAbstractItemView.SingleSelection)
        self.elec_list.setMinimumWidth(150)
        self.set_elec_names()

        main_hbox = QHBoxLayout()
        main_hbox.addWidget(self.plt)
        main_hbox.addWidget(self.elec_list)

        vbox = QVBoxLayout()
        vbox.addLayout(button_hbox)
        vbox.addLayout(slider_hbox)
        vbox.addLayout(main_hbox)

        central_widget = QWidget()
        central_widget.setLayout(vbox)
        self.setCentralWidget(central_widget)

        name = self.get_current_elec()
        if name:
            self.update_group_color()
        if name in self.elec_matrix:
            self.move_cursors_to_pos()

    def load_image_data(self):
        # Specified resolution freesurfer uses, images must be
        # This resolution so we can plot them overlaid
        vx, vy, vz = VOXEL_SIZES

        # prepare MRI data
        self.img_data = load_image_data('mri', 'brain.mgz')
        self.mri_min = self.img_data.min()
        self.mri_max = self.img_data.max()
        # voxel_sizes = nib.affines.voxel_sizes(img.affine)
        nx, ny, nz = np.array(self.img_data.shape, dtype='float')
        # check mri size to make sure it's correct
        if nx != vx or ny != vx or nz != vz:
            raise ValueError(f'MRI dimensions found {nx} {ny} {nz} '
                             f'expected dimensions were {vx} {vy} {vz}, '
                             'check recon and contact developers')
        # ready ct
        self.ct_data = load_image_data('CT', 'rCT.nii', 'coreg_CT_MR')
        self.ct_min = np.nanmin(self.ct_data)
        self.ct_max = np.nanmax(self.ct_data)
        cx, cy, cz = np.array(self.ct_data.shape, dtype='float')
        # check CT size to make sure it's correct
        if cx != vx or cy != vx or cz != vz:
            raise ValueError(f'CT dimensions found {cx} {cy} {cz} '
                             f'expected dimensions were {vx} {vy} {vz}, '
                             'make sure `img_pipe.coreg_CT_MR` was run '
                             'even if the alignment was done by hand '
                             '(it resamples the CT to the right size)')

        # prepare pial data
        self.pial_data = dict()
        for hemi in ('lh', 'rh'):
            self.pial_data[hemi] = load_image_data(
                'surf', f'{hemi}.pial.filled.mgz', 'label')
            self.pial_data[hemi] = binary_closing(self.pial_data[hemi])

        # This is the current slice for indexing (as integers for indexing)
        self.current_slice = np.array([vx / 2, vy / 2, vz / 2], dtype=np.int)

    def get_button_bar(self):
        button_hbox = QHBoxLayout()

        new_button = QPushButton('New')
        button_hbox.addWidget(new_button)
        new_button.released.connect(self.new_elec)

        button_hbox.addStretch(4)

        prev_button = QPushButton('Prev')
        button_hbox.addWidget(prev_button)
        prev_button.released.connect(self.previous_elec)

        self.elec_label = QLabel(self.get_current_elec())
        button_hbox.addWidget(self.elec_label)

        next_button = QPushButton('Next')
        button_hbox.addWidget(next_button)
        next_button.released.connect(self.next_elec)

        button_hbox.addStretch(1)

        RAS_label = QLabel('RAS=')
        self.RAS_textbox = QPlainTextEdit(
            '{:.2f}, {:.2f}, {:.2f}'.format(*self.cursors_to_RAS()))
        self.RAS_textbox.setMaximumHeight(25)
        self.RAS_textbox.setMaximumWidth(200)
        self.RAS_textbox.focusOutEvent = self.update_RAS
        self.RAS_textbox.textChanged.connect(self.check_update_RAS)
        button_hbox.addWidget(RAS_label)
        button_hbox.addWidget(self.RAS_textbox)

        mark_button = QPushButton('Mark')
        button_hbox.addWidget(mark_button)
        mark_button.released.connect(self.mark_elec)

        remove_button = QPushButton('Remove')
        button_hbox.addWidget(remove_button)
        remove_button.released.connect(self.remove_elec)

        self.group_selector = ComboBox()
        self.group_selector.addItem('auto')
        group_model = self.group_selector.model()

        for i in range(MAX_N_GROUPS):
            self.group_selector.addItem(' ')
            rgba = cm.ScalarMappable(
                norm=NORM, cmap=ELECTRODE_COLORS).to_rgba(i)
            color = QtGui.QColor()
            color.setRgb(*[c * 255 for c in rgba])
            brush = QtGui.QBrush(color)
            brush.setStyle(QtCore.Qt.SolidPattern)
            group_model.setData(group_model.index(i + 1, 0),
                                brush, QtCore.Qt.BackgroundRole)
        self.group_selector.currentIndexChanged.connect(
            self.update_group_color)
        self.group_selector.clicked.connect(self.set_auto_color)
        button_hbox.addWidget(self.group_selector)

        return button_hbox

    def get_slider_bar(self):

        def make_label(name):
            label = QLabel(name)
            label.setAlignment(QtCore.Qt.AlignCenter)
            return label

        def make_slider(smin, smax, sval, sfun):
            slider = QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(smin)
            slider.setMaximum(smax)
            slider.setValue(sval)
            slider.valueChanged.connect(sfun)
            slider.keyPressEvent = self.keyPressEvent
            return slider

        slider_hbox = QHBoxLayout()

        mri_vbox = QVBoxLayout()
        mri_vbox.addWidget(make_label('MRI min'))
        mri_vbox.addWidget(make_label('MRI max'))
        slider_hbox.addLayout(mri_vbox)

        mri_slider_vbox = QVBoxLayout()
        self.mri_min_slider = make_slider(
            self.mri_min, self.mri_max, self.mri_min, self.update_mri_min)
        mri_slider_vbox.addWidget(self.mri_min_slider)
        self.mri_max_slider = make_slider(
            self.mri_min, self.mri_max, self.mri_max, self.update_mri_max)
        mri_slider_vbox.addWidget(self.mri_max_slider)
        slider_hbox.addLayout(mri_slider_vbox)

        ct_vbox = QVBoxLayout()
        ct_vbox.addWidget(make_label('CT min'))
        ct_vbox.addWidget(make_label('CT max'))
        slider_hbox.addLayout(ct_vbox)

        ct_slider_vbox = QVBoxLayout()
        self.ct_min_slider = make_slider(self.ct_min, self.ct_max,
                                         self.ct_min, self.update_ct_min)
        ct_slider_vbox.addWidget(self.ct_min_slider)
        self.ct_max_slider = make_slider(self.ct_min, self.ct_max,
                                         self.ct_max, self.update_ct_max)
        ct_slider_vbox.addWidget(self.ct_max_slider)
        slider_hbox.addLayout(ct_slider_vbox)

        radius_slider_vbox = QVBoxLayout()
        radius_slider_vbox.addWidget(make_label('radius'))
        elec_max = int(np.mean(ELEC_PLOT_SIZE) // 50)
        self.radius_slider = make_slider(0, elec_max, self.elec_radius,
                                         self.update_radius)
        radius_slider_vbox.addWidget(self.radius_slider)
        slider_hbox.addLayout(radius_slider_vbox)
        return slider_hbox

    def make_elec_image(self, axis):
        """Make electrode data higher resolution so it looks better."""
        elec_image = np.zeros(ELEC_PLOT_SIZE) + np.nan
        vx, vy, vz = VOXEL_SIZES

        def color_elec_radius(elec_image, xf, yf, group):
            '''Take the fraction across each dimension of the RAS
               coordinates converted to xyz and put a circle in that
               position in this larger resolution image.'''
            ex, ey = np.round(np.array([xf, yf]) * ELEC_PLOT_SIZE).astype(int)
            for i in range(-self.elec_radius, self.elec_radius + 1):
                for j in range(-self.elec_radius, self.elec_radius + 1):
                    if (i**2 + j**2)**0.5 < self.elec_radius:
                        # negative y because y axis is inverted
                        elec_image[-(ey + i), ex + j] = group
            return elec_image

        for name in self.elec_matrix:
            # move from middle-centered (half coords positive, half negative)
            # to bottom-left corner centered (all coords positive).
            xyz = self.RAS_to_cursors(name)
            # check if closest to that voxel
            if np.round(xyz[axis]).astype(int) == self.current_slice[axis]:
                x, y, z = xyz
                if axis == 0:
                    elec_image = color_elec_radius(
                        elec_image, y / vy, z / vz,
                        self.elec_matrix[name][3])  # group
                elif axis == 1:
                    elec_image = color_elec_radius(
                        elec_image, x / vx, z / vx,
                        self.elec_matrix[name][3])  # group
                elif axis == 2:
                    elec_image = color_elec_radius(
                        elec_image, x / vx, y / vy,
                        self.elec_matrix[name][3])  # group
        return elec_image

    def make_slice_plots(self):
        self.plt = SlicePlots(self)
        # Plot sagittal (0), coronal (1) or axial (2) view
        self.images = dict(mri=list(), ct=dict(), elec=dict(),
                           pial=dict(lh=list(), rh=list()),
                           cursor=dict(), cursor2=dict())

        vx, vy, vz = VOXEL_SIZES
        im_ranges = [[0, vy, 0, vz],
                     [0, vx, 0, vz],
                     [0, vx, 0, vy]]
        im_labels = [['Inferior', 'Posterior'],
                     ['Inferior', 'Left'],
                     ['Posterior', 'Left']]
        for axis in range(3):
            img_data = np.take(self.img_data, self.current_slice[axis],
                               axis=axis).T
            self.images['mri'].append(self.plt.axes[0, axis].imshow(
                img_data, cmap=cm.gray, aspect='auto'))
            ct_data = np.take(self.ct_data, self.current_slice[axis],
                              axis=axis).T
            self.images['ct'][(0, axis)] = self.plt.axes[0, axis].imshow(
                ct_data, cmap=cm.hot, aspect='auto', alpha=0.5,
                vmin=CT_MIN_VAL, vmax=np.nanmax(self.ct_data))
            self.images['ct'][(1, axis)] = self.plt.axes[1, axis].imshow(
                ct_data, cmap=cm.gray, aspect='auto')
            for hemi in ('lh', 'rh'):
                pial_data = np.take(self.pial_data[hemi],
                                    self.current_slice[axis],
                                    axis=axis).T
                self.images['pial'][hemi].append(
                    self.plt.axes[0, axis].contour(
                        pial_data, linewidths=0.5, colors='y'))
            for axis2 in range(2):
                self.images['elec'][(axis2, axis)] = \
                    self.plt.axes[axis2, axis].imshow(
                    self.make_elec_image(axis), cmap=ELECTRODE_COLORS,
                    aspect='auto', extent=im_ranges[axis],
                    alpha=1, vmin=0, vmax=MAX_N_GROUPS)
                self.images['cursor'][(axis2, axis)] = \
                    self.plt.axes[axis2, axis].plot(
                    (self.current_slice[1], self.current_slice[1]),
                    (0, VOXEL_SIZES[axis]), color=[0, 1, 0], linewidth=0.25)[0]
                self.images['cursor2'][(axis2, axis)] = \
                    self.plt.axes[axis2, axis].plot(
                    (0, VOXEL_SIZES[axis]), color=[0, 1, 0], linewidth=0.25)[0]
                self.plt.axes[axis2, axis].set_facecolor('k')

                self.plt.axes[axis2, axis].invert_yaxis()
                self.plt.axes[axis2, axis].set_xticks([])
                self.plt.axes[axis2, axis].set_yticks([])

                self.plt.axes[axis2, axis].set_xlabel(im_labels[axis][0],
                                                      fontsize=4)
                self.plt.axes[axis2, axis].set_ylabel(im_labels[axis][1],
                                                      fontsize=4)
                self.plt.axes[axis2, axis].axis(im_ranges[axis])
        self.plt.fig.suptitle('', fontsize=14)
        self.plt.fig.tight_layout()
        self.plt.fig.subplots_adjust(left=0.03, bottom=0.07,
                                     wspace=0.15, hspace=0.15)
        self.plt.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.plt.fig.canvas.mpl_connect('button_release_event', self.on_click)

    def set_elec_names(self):
        self.elec_list_model = QtGui.QStandardItemModel(self.elec_list)
        for name in self.elec_names:
            self.elec_list_model.appendRow(QtGui.QStandardItem(name))
        for name in self.elec_matrix:
            self.color_list_item(name=name)
        self.elec_list.setModel(self.elec_list_model)
        self.elec_list.clicked.connect(self.change_elec)
        if self.elec_names or self.elec_matrix:
            self.elec_list.setCurrentIndex(
                self.elec_list_model.index(self.elec_index, 0))
        self.elec_list.keyPressEvent = self.keyPressEvent

    def set_auto_color(self):
        group = self.auto_group()
        self.color_group_selector(group)

    def update_group_color(self):
        group = self.get_group()
        self.color_group_selector(group)

    def color_group_selector(self, group):
        rgb = cm.ScalarMappable(
            norm=NORM, cmap=ELECTRODE_COLORS).to_rgba(group)[:3]
        rgb = (255 * np.array(rgb)).round().astype(int)
        self.group_selector.setStyleSheet(
            'background-color: rgb({:d},{:d},{:d})'.format(*rgb))

    def on_scroll(self, event):
        """Process mouse scroll wheel event to zoom."""
        self.zoom(event.step, draw=True)

    def zoom(self, sign=1, draw=False):
        """Zoom in on the image."""
        delta = ZOOM_STEP_SIZE * sign
        for axis in range(3):
            for axis2 in range(2):
                ax = self.plt.axes[axis2, axis]
                xmid = self.images['cursor'][(axis2, axis)].get_xdata()[0]
                ymid = self.images['cursor2'][(axis2, axis)].get_ydata()[0]
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                xwidth = (xmax - xmin) / 2 - delta
                ywidth = (ymax - ymin) / 2 - delta
                if xwidth <= 0 or ywidth <= 0:
                    return
                ax.set_xlim(xmid - xwidth, xmid + xwidth)
                ax.set_ylim(ymid - ywidth, ymid + ywidth)
                self.images['cursor'][(axis2, axis)].set_ydata([ymin, ymax])
                self.images['cursor2'][(axis2, axis)].set_xdata(
                    [xmin, xmax])
        if draw:
            self.plt.fig.canvas.draw()

    def get_current_elec(self, name=None):
        if name is None:
            return self.elec_names[self.elec_index] if self.elec_names else ''
        else:
            return name

    def update_elec_selection(self):
        name = self.get_current_elec()
        self.elec_label.setText(name)
        if name:
            self.elec_list.setCurrentIndex(
                self.elec_list_model.index(self.elec_index, 0))
        if name in self.elec_matrix:
            self.move_cursors_to_pos()
        self.update_group_color()
        self.plt.fig.canvas.draw()

    def change_elec(self, index):
        """Change current electrode to the item selected."""
        self.elec_index = index.row()
        self.update_elec_selection()

    @pyqtSlot()
    def previous_elec(self):
        if len(self.elec_names) > 0:
            self.elec_index = (self.elec_index - 1) % len(self.elec_names)
        self.update_elec_selection()

    @pyqtSlot()
    def next_elec(self):
        if len(self.elec_names) > 0:
            self.elec_index = (self.elec_index + 1) % len(self.elec_names)
        self.update_elec_selection()

    @pyqtSlot()
    def new_elec(self):
        name = None
        while name is None:
            name, ok = QInputDialog.getText(self, 'Input dialog',
                                            'Enter name:')
            if ok:
                name = str(name)
            if name in self.elec_names:
                name = None
                QMessageBox('Electrode name already exists, '
                            'to overwrite, just press mark '
                            'with that electrode selected')
        self.elec_names.append(name)
        self.set_elec_names()

    @pyqtSlot()
    def update_RAS(self, event):
        text = self.RAS_textbox.toPlainText().replace('\n', '')
        xyz = text.split(',')
        if len(xyz) != 3:
            xyz = text.split(' ')  # spaces also okay as in freesurfer
        xyz = [var.lstrip().rstrip() for var in xyz]

        def reset_RAS_label():
            self.RAS_textbox.setPlainText(  # put back if not numeric
                '{:.2f}, {:.2f}, {:.2f}'.format(*self.cursors_to_RAS()))

        if len(xyz) != 3:
            reset_RAS_label()
            return
        all_float = all([all([dig.isdigit() or dig in ('-', '.')
                              for dig in var]) for var in xyz])
        if not all_float:
            reset_RAS_label()
            return

        xyz = np.array([float(var) for var in xyz])
        wrong_size = any([var < -n / 2 or var > n / 2 for var, n in
                          zip(xyz, VOXEL_SIZES)])
        if wrong_size:
            reset_RAS_label()
            return

        self.move_cursors_to_pos(xyz=xyz + VOXEL_SIZES // 2)

    @pyqtSlot()
    def check_update_RAS(self):
        if '\n' in self.RAS_textbox.toPlainText():
            self.update_RAS(event=None)

    def get_group(self):
        group = self.group_selector.currentIndex()
        if group == 0:  # auto
            group = self.auto_group()
        else:
            group -= 1  # auto is first
        return group

    def color_list_item(self, name=None, clear=False):
        """Color the item in the view list for easy id of marked."""
        name = self.get_current_elec(name=name)
        color = QtGui.QColor('white')
        if not clear:
            # we need the normalized color map
            group = self.elec_matrix[name][3]
            rgb = cm.ScalarMappable(
                norm=NORM, cmap=ELECTRODE_COLORS).to_rgba(group)[:3]
            color.setRgb(*[c * 255 for c in rgb])
        brush = QtGui.QBrush(color)
        brush.setStyle(QtCore.Qt.SolidPattern)
        self.elec_list_model.setData(
            self.elec_list_model.index(self.elec_names.index(name), 0),
            brush, QtCore.Qt.BackgroundRole)

    def auto_group(self):
        name = self.get_current_elec()
        if not name:
            return 0
        group = 0
        sorted_names = sorted(self.elec_names)

        def find_begin_end_numbers(name):
            name = list(name)
            numbers = ''
            while name[0].isdigit():
                numbers += name.pop(0)
            while name[-1].isdigit():
                numbers = name.pop(-1) + numbers
            return numbers

        pattern = sorted_names[0].replace(
            find_begin_end_numbers(sorted_names[0]), '')  # fencepost
        for this_name in sorted(self.elec_names):
            this_pattern = this_name.replace(
                find_begin_end_numbers(this_name), '')
            if this_pattern != pattern:
                pattern = this_pattern
                group += 1
            if this_name == name:
                return group
        return -1

    @pyqtSlot()
    def mark_elec(self):
        """Mark the current electrode as at the current crosshair point."""
        self.remove_elec()
        name = self.get_current_elec()
        if name:
            self.elec_matrix[name] = \
                np.append(self.cursors_to_RAS(), self.get_group(), 'n/a')
            self.color_list_item()
            self.update_elec_images(draw=True)
            save_electrodes()
            self.next_elec()

    @pyqtSlot()
    def remove_elec(self):
        name = self.get_current_elec()
        if name in self.elec_matrix:
            self.color_list_item(clear=True)
            self.elec_matrix.pop(name)
            save_electrodes()
            self.update_elec_images(draw=True)

    def update_elec_images(self, axis_selected=None, draw=False):
        for axis in range(3) if axis_selected is None else [axis_selected]:
            for axis2 in range(2):
                self.images['elec'][(axis2, axis)].set_data(
                    self.make_elec_image(axis))
        if draw:
            self.plt.fig.canvas.draw()

    def update_mri_images(self, axis_selected=None, draw=False):
        for axis in range(3) if axis_selected is None else [axis_selected]:
            if self.T1_on:
                img_data = np.take(self.img_data, self.current_slice[axis],
                                   axis=axis).T
            else:
                img_data = np.take(np.zeros(VOXEL_SIZES),
                                   self.current_slice[axis], axis=axis).T
            self.images['mri'][axis].set_data(img_data)
        if draw:
            self.plt.fig.canvas.draw()

    def update_ct_images(self, axis_selected=None, draw=False):
        for axis in range(3) if axis_selected is None else [axis_selected]:
            ct_data = np.take(self.ct_data, self.current_slice[axis],
                              axis=axis).T
            # Threshold the CT so only bright objects (electrodes) are visible
            ct_data[ct_data < self.ct_min] = np.nan
            ct_data[ct_data > self.ct_max] = np.nan
            for axis2 in range(2):
                self.images['ct'][(axis2, axis)].set_data(ct_data)
        if draw:
            self.plt.fig.canvas.draw()

    def update_pial_images(self, axis_selected=None, draw=False):
        for axis in range(3) if axis_selected is None else [axis_selected]:
            for hemi in ('lh', 'rh'):
                for collection in self.images['pial'][hemi][axis].collections:
                    collection.remove()
                self.images['pial'][hemi][axis].collections = list()
                if self.pial_surf_on:
                    pial_data = np.take(self.pial_data[hemi],
                                        self.current_slice[axis],
                                        axis=axis).T
                    self.images['pial'][hemi][axis] = \
                        self.plt.axes[0, axis].contour(
                            pial_data, linewidths=0.5, colors='y')
        if draw:
            self.plt.fig.canvas.draw()

    def update_images(self, axis=None, draw=True):
        self.update_mri_images(axis_selected=axis)
        self.update_ct_images(axis_selected=axis)
        self.update_pial_images(axis_selected=axis)
        self.update_elec_images(axis_selected=axis)
        if draw:
            self.plt.fig.canvas.draw()

    def update_mri_min(self):
        """Update MRI min slider value."""
        if self.mri_min_slider.value() > self.mri_max:
            tmp = self.mri_max
            self.mri_max = self.mri_min_slider.value()
            self.mri_max_slider.setValue(self.mri_max)
            self.mri_min = tmp
            self.mri_min_slider.setValue(self.mri_min)
        else:
            self.mri_min = self.mri_min_slider.value()
        self.update_mri_scale()

    def update_mri_max(self):
        """Update MRI max slider value."""
        if self.mri_max_slider.value() < self.mri_min:
            tmp = self.mri_min
            self.mri_min = self.mri_max_slider.value()
            self.mri_min_slider.setValue(self.mri_min)
            self.mri_max = tmp
            self.mri_max_slider.setValue(self.mri_max)
        else:
            self.mri_max = self.mri_max_slider.value()
        self.update_mri_scale()

    def update_mri_scale(self):
        for mri_img in self.images['mri']:
            mri_img.set_clim([self.mri_min, self.mri_max])
        self.plt.fig.canvas.draw()

    def update_ct_min(self):
        """Update CT min slider value."""
        if self.ct_min_slider.value() > self.ct_max:
            tmp = self.ct_max
            self.ct_max = self.ct_min_slider.value()
            self.ct_max_slider.setValue(self.ct_max)
            self.ct_min = tmp
            self.ct_min_slider.setValue(self.ct_min)
        else:
            self.ct_min = self.ct_min_slider.value()
        self.update_ct_images(draw=True)

    def update_ct_max(self):
        """Update CT max slider value."""
        if self.ct_max_slider.value() < self.ct_min:
            tmp = self.ct_min
            self.ct_min = self.ct_max_slider.value()
            self.ct_min_slider.setValue(self.ct_min)
            self.ct_max = tmp
            self.ct_max_slider.setValue(self.ct_max)
        else:
            self.ct_max = self.ct_max_slider.value()
        self.update_ct_images(draw=True)

    def update_radius(self):
        """Update electrode radius."""
        self.elec_radius = np.round(self.radius_slider.value()).astype(int)
        self.update_elec_images(draw=True)

    def get_axis_selected(self, x, y, return_pos=False):
        """Get which axis was clicked."""
        def get_position(fxy, axis, b_box):
            """Helper to determine where on axis was clicked."""
            ax = self.plt.axes[0, axis]
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            fx, fy = fxy
            pos = [((fx - b_box.xmin) / b_box.width),
                   ((fy - b_box.ymin) / b_box.height)]
            pos[0] = pos[0] * (xmax - xmin) + xmin
            pos[1] = pos[1] * (ymax - ymin) + ymin
            return pos
        fxy = self.plt.fig.transFigure.inverted().transform((x, y))
        for axis in range(3):
            for axis2 in range(2):
                b_box = self.plt.axes[axis2, axis].get_position()
                if b_box.contains(*fxy):
                    if return_pos:
                        return axis, get_position(fxy, axis, b_box)
                    else:
                        return axis
        if return_pos:
            return None, None
        else:
            return None

    def move_cursors_to_pos(self, xyz=None):
        if xyz is None:
            xyz = self.RAS_to_cursors()
        self.current_slice = np.array([*xyz]).round().astype(int)
        x, y, z = xyz
        self.move_cursor_to(0, x=y, y=z)
        self.move_cursor_to(1, x=x, y=z)
        self.move_cursor_to(2, x=x, y=y)
        self.zoom(0)  # dofesn't actually zoom just resets view to center
        self.update_images(draw=True)
        self.update_RAS_label()

    def move_cursor_to(self, axis, x=None, y=None):
        for axis2 in range(2):
            if x is None:
                x = self.images['cursor'][(axis2, axis)].get_xdata()[0]
            if y is None:
                y = self.images['cursor2'][(axis2, axis)].get_ydata()[0]
            self.images['cursor2'][(axis2, axis)].set_ydata([y, y])
            self.images['cursor'][(axis2, axis)].set_xdata([x, x])
        self.zoom(0)

    def move_cursor(self, axis, sign, ori):
        for axis2 in range(2):
            if ori == 'h':
                (xmin, xmax), (ymin, ymax) = \
                    self.images['cursor2'][(axis2, axis)].get_data()
                self.images['cursor2'][(axis2, axis)].set_ydata(
                    [ymin + sign, ymax + sign])
            elif ori == 'v':
                (xmin, xmax), (ymin, ymax) = \
                    self.images['cursor'][(axis2, axis)].get_data()
                self.images['cursor'][(axis2, axis)].set_xdata(
                    [xmin + sign, xmax + sign])
        self.zoom(0)  # center cursor

    def keyPressEvent(self, event):
        """Executes when the user presses a key.

        Electrode adding:
        ----
        n: enter the name of a new device (e.g. 'frontalgrid',
                                           'hippocampaldepth')
        e: insert an electrode at the current green crosshair position
        u: remove electrode at the current crosshair position
           (can be thought of like 'undo')
        """
        if event.key() == 'escape':
            self.close()

        if event.text() == 't':
            # Toggle pial surface outline on and off
            self.pial_surf_on = not self.pial_surf_on
            self.update_pial_images(draw=True)

        if event.text() == 'b':
            # Toggle T1 scan on and off
            self.T1_on = not self.T1_on
            self.update_mri_images(draw=True)

        if event.text() == 'n':
            self.new_elec()

        if event.text() == 'c':
            self.zoom(0, draw=True)

        if event.text() == 'h':
            # Show help
            QMessageBox.information(
                self, 'Help',
                "Help: 'n': name device, 'e': mark electrode, "
                "'u': remove electrode, 't': toggle pial, "
                "'c': center view on cursors, "
                "b': toggle brain, '3': show 3D view\n"
                "Maximum intensity projection views: "
                "'+'/'-': zoom, left/right arrow: left/right "
                "up/down arrow: superior/inferior "
                "page up/page down arrow: anterior/posterior")

        if event.text() == 'e':
            self.mark_elec()

        if event.text() == 'u':
            self.remove_elec()

        if event.text() == '3':
            self.launch_3D_viewer()

        if event.text() in ('=', '+', '-'):
            self.zoom(sign=-2 * (event.text() == '-') + 1, draw=True)

        # Changing slices
        if event.key() in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down,
                           QtCore.Qt.Key_Left, QtCore.Qt.Key_Right,
                           QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown):
            sign = (2 * (event.key() in (QtCore.Qt.Key_Up,
                                         QtCore.Qt.Key_Right,
                                         QtCore.Qt.Key_PageUp)) - 1)
            if event.key() in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
                self.current_slice[2] += sign
                self.move_cursor(axis=0, sign=sign, ori='h')
                self.move_cursor(axis=1, sign=sign, ori='h')
                self.update_images(axis=2)
            elif event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
                self.current_slice[0] += sign
                self.move_cursor(axis=1, sign=sign, ori='v')
                self.move_cursor(axis=2, sign=sign, ori='v')
                self.update_images(axis=0)
            elif event.key() in (QtCore.Qt.Key_PageUp,
                                 QtCore.Qt.Key_PageDown):
                self.current_slice[1] += sign
                self.move_cursor(axis=0, sign=sign, ori='v')
                self.move_cursor(axis=2, sign=sign, ori='h')
                self.update_images(axis=1)

    def on_click(self, event):
        """Executes on mouse click events

        Moves appropriate subplot axes to (x,y,z) view on MRI and CT views.
        """
        # Transform coordinates to figure coordinates

        axis_selected, pos = self.get_axis_selected(
            event.x, event.y, return_pos=True)

        if axis_selected is not None and pos is not None:
            x, y = pos
            self.move_cursor_to(axis_selected, x, y)
            if axis_selected == 0:
                self.current_slice[1] = np.round(x).astype(int)
                self.move_cursor_to(1, y=y)
                self.current_slice[2] = np.round(y).astype(int)
                self.move_cursor_to(2, y=x)
            elif axis_selected == 1:
                self.current_slice[0] = np.round(x).astype(int)
                self.move_cursor_to(0, y=y)
                self.current_slice[2] = np.round(y).astype(int)
                self.move_cursor_to(2, x=x)
            elif axis_selected == 2:
                self.current_slice[0] = np.round(x).astype(int)
                self.move_cursor_to(0, x=y)
                self.current_slice[1] = np.round(y).astype(int)
                self.move_cursor_to(1, x=x)
            non_selected = list(set(range(3)).difference(set([axis_selected])))
            self.update_images(axis=non_selected[0], draw=False)
            self.update_images(axis=non_selected[1], draw=True)
            self.update_RAS_label()

    def update_RAS_label(self):
        self.RAS_textbox.setPlainText('{:.2f}, {:.2f}, {:.2f}'.format(
            *self.cursors_to_RAS()))

    def cursors_to_RAS(self):
        """Convert slice coordinate from the viewer to surface RAS
        Returns
        -------
        elec : np.array
            RAS coordinate of the requested slice coordinate
        """
        return (np.array([self.images['cursor'][(0, 1)].get_xdata()[0],
                          self.images['cursor'][(0, 0)].get_xdata()[0],
                          self.images['cursor2'][(0, 0)].get_ydata()[0]]
                         ) - VOXEL_SIZES // 2)

    def RAS_to_cursors(self, name=None):
        """Convert acpc-zero-centered RAS to slice indices.
        Returns
        -------
        slice : np.array
            The slice coordinates of the given RAS data
        """
        name = self.get_current_elec(name=name)
        return np.array(self.elec_matrix[name][:3]) + VOXEL_SIZES // 2

    def launch_3D_viewer(self):
        """Launch 3D viewer to visualize electrodes."""
        # Get appropriate hemisphere
        plot_brain(get_rois(), opacity=0.3)

        # Plot the electrodes we have so far
        coords = list()
        colors = list()
        names = list()
        for name in self.elec_matrix:
            R, A, S, group, _ = self.elec_matrix[name]
            coords.append([R, A, S])
            colors.append(cm.ScalarMappable(
                norm=NORM, cmap=ELECTRODE_COLORS).to_rgba(group)[:3])
            names.append(name)
        add_electrodes(np.array(coords), color=np.array(colors),
                       msize=4, labels=names)


def launch_electrode_picker():
    """Wrapper for the ElcetrodePicker object."""
    app = QApplication(['Electrode Picker'])
    electrode_picker = ElectrodePicker()
    electrode_picker.show()
    app.exec_()