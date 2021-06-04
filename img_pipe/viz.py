"""Vizualization functions for img_pipe."""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD (3-clause)
import os
import os.path as op

import numpy as np
from scipy.ndimage import binary_closing

from img_pipe.config import (VOXEL_SIZES, IMG_RANGES, IMG_LABELS,
                             CT_MIN_VAL, MAX_N_GROUPS, UNIQUE_COLORS, N_COLORS,
                             SUBCORTICAL_INDICES, ZOOM_STEP_SIZE,
                             ELEC_PLOT_SIZE, CORTICAL_SURFACES, ELECTRODE_CMAP)
from img_pipe.utils import (check_fs_vars, check_dir, get_fs_labels,
                            get_fs_colors, load_electrode_names,
                            load_electrodes, save_electrodes, load_image_data)
#                            get_device_names)
import mne  # noqa

import matplotlib as mpl
import matplotlib.pyplot as plt  # noqa
from matplotlib import cm  # noqa
import matplotlib.colors as mcolors  # noqa
from matplotlib.widgets import Slider, Button  # noqa

from PyQt5 import QtCore, QtGui, Qt  # noqa
from PyQt5.QtCore import pyqtSlot  # noqa
from PyQt5.QtWidgets import (QApplication, QMainWindow,  # noqa
                             QVBoxLayout, QHBoxLayout, QLabel,
                             QInputDialog, QMessageBox, QWidget,
                             QListView, QSlider, QPushButton,
                             QComboBox, QPlainTextEdit)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # noqa


class ROI:
    def __init__(self, name, color=None, opacity=1.0, atlas='desikan-killiany',
                 template=None, representation=None):
        """Class defining a region of interest (ROI) mesh.

        This could be, for example, a mesh for the left hippocampus,
        or a mesh for the right superior temporal gyrus.

        Parameters
        ----------
        name: str
            The name of the ROI (e.g 'lh.pial'). See :func:`img_pipe.list_rois`
            for the full list.
        color: tuple
            Tuple for the ROI's color where each value is between 0.0 and 1.0.
        opacity: float (between 0.0 and 1.0)
            opacity of the mesh, between 0.0 and 1.0
        atlas: str
            The atlas parcellation; 'desikan-killiany', 'DKT' or 'destrieux'.
        template: str
            Name of the template to use if plotting electrodes on a
            group template brain. e.g. 'cvs_avg35_inMNI152'.
        representation: str
            The representation of the volume in 3D plotting; 'surface' or
            'wireframe'. The default is 'surface'.

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

        # handle number indexed names
        if isinstance(name, int):
            if name not in number_dict:
                raise ValueError(f'Name {name} not recognized')
            name = number_dict[name]
        if name in CORTICAL_SURFACES:
            if template is None:
                roi_dir = check_dir(op.join(base_path, 'surf'),
                                    'img_pipe.recon')
            else:
                roi_dir = check_dir(op.join(os.environ['FREESURFER_HOME'],
                                            'subjects', template, 'surf'),
                                    instructions='Check your freesurfer '
                                                 'installation')
            name = CORTICAL_SURFACES[name]
        else:
            if template is None:
                roi_dir = check_dir(op.join(base_path, 'label', atlas),
                                    'img_pipe.label')
            else:
                roi_dir = check_dir(op.join(os.environ['FREESURFER_HOME'],
                                            'subjects', template, 'label',
                                            atlas), 'img_pipe.warp')
            if name not in os.listdir(roi_dir):
                raise ValueError(f'Name {name} not recongized')
            # change number label to name
            if isinstance(name, int):
                name = number_dict[name]
        self.name = name
        self.vert, self.tri = mne.read_surface(op.join(roi_dir, name))
        if color is None:
            self.color = tuple((c / 255 for c in color_dict[name])
                               if name in color_dict else (0.8,) * 3)
        else:
            self.color = color
        self.opacity = opacity
        self.representation = \
            'surface' if representation is None else representation


def get_rois(group='all', template=None, opacity=1.0,
             representation='surface'):
    """Get the subcortial regions of interest

    Parameters
    ----------
    group: str
        - 'all' for pial and subcortial structures
        - 'pial' for just pial surfaces
        - 'subcortical' for just subcortical structures
        - 'inflated' for left and right inflated
        - 'white' for left and right white matter
    template: str
        Name of the template to use if plotting electrodes on a
        group template brain. e.g. 'cvs_avg35_inMNI152'.
    opacity: float
        The opacity of the mesh, between 0.0 and 1.0.
    rep: str
        The representation type of the 3D mesh; 'surface' or 'wireframe'.
    """
    if group in ('all', 'pial', 'inflated', 'white'):
        name = 'Pial' if group == 'all' else group.capitalize()
        cortex = [ROI(f'{hemi}-{name}', opacity=opacity,
                      representation=representation)
                  for hemi in ('Left', 'Right')]
    if group in ('all', 'subcortical'):
        subcortical = \
            [ROI(idx, opacity=opacity, representation=representation)
             for idx in SUBCORTICAL_INDICES]
    if group == 'all':
        return cortex + subcortical
    elif group == 'subcortical':
        return subcortical
    elif group in ('pial', 'inflated', 'white'):
        return cortex
    else:
        raise ValueError(f'Unrecognized group {group}')


def plot_brain(rois=None, picks=None, elec_scale=5, distance=500,
               azimuth=None, elevation=None, opacity=1.0, show=True,
               verbose=True):
    """Plots multiple meshes on one figure.
    Defaults to plotting both hemispheres of the pial surface.

    Parameters
    ----------
    rois: list
        List of roi objects created like so:
        hipp_roi = ROI(name='Left-Hippocampus', color=(0.5, 0.1, 0.8)))
        See :func:`img_pipe.list_rois` for available ROI names.
        Defaults to [ROI('Left-Pial'), ROI('Right-Pial')]
    picks: list
        If None, all electrodes are plotted else only picks are plotted.
    elec_scale: float
        How large to plot the electrodes (mm).
    azimuth: float
        Azimuth for brain view.
    elevation: float
        Elevation for brain view.
    opacity: float
        How transparent to make the image between 0 (transparent) and 1 (not).
    show: bool
        whether to show the figure in an interactive window
    verbose : bool
        Whether to print text updating on the status of the function.

    Returns
    -------
    renderer.figure: mayavi.core.scene.Scene
        The scene for figure handling such as taking images.

    Example
    -------
    >>> from img_pipe import ROI, plot_brain
    >>> pial = ROI('Left-Pial', (0.6, 0.3, 0.6), opacity=0.1,
    >>>            representation='wireframe')
    >>> hipp = ROI('Left-Hippocampus', (0.5, 0.1, 0.8), opacity=1.0)
    >>> plot_brain(rois=[pial, hipp])
    """
    elecs = load_electrodes()
    if picks is not None:
        elecs = {ch: elecs[ch] for ch in picks if ch in elecs}
    if rois is None:
        rois = get_rois('pial', opacity=opacity)
    renderer = mne.viz.backends.renderer.create_3d_figure(
        size=(1200, 900), bgcolor='w', scene=False)
    mne.viz.set_3d_view(figure=renderer.figure, distance=distance,
                        azimuth=azimuth, elevation=elevation)
    for elec_data in elecs.values():
        x, y, z, group, _ = elec_data
        renderer.sphere(center=(x, y, z), color=ELECTRODE_CMAP(group)[:3],
                        scale=elec_scale)
    for roi in rois:
        renderer.mesh(*roi.vert.T, triangles=roi.tri, color=roi.color,
                      opacity=roi.opacity, representation=roi.representation)
    if show:
        renderer.show()
    return renderer.figure


'''
class ElectrodeGUI(QMainWindow):
    """Pick electrodes manually using a coregistered MRI and CT."""

    def __init__(self, verbose=True):
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

        """
        # initialize QMainWindow class
        super(ElectrodeGUI, self).__init__()

        # load imaging data
        self.base_path = check_fs_vars()

        # load CT
        self.ct_data = load_image_data('CT', 'rCT.nii', 'coreg_CT_MR',
                                       reorient=True)  # convert to ras)

        # get names of contacts to localize
        self.elec_names = load_electrode_names()

        self.devices = self.get_devices()

        # initialize electrode data
        self.elec_index = 0

        # add already marked electrodes if they exist
        self.elecs = load_electrodes()
        for name in self.elecs:
            if name not in self.elec_names:
                self.elec_names.append(name)

        # GUI design
        self.setWindowTitle('Electrode GUI')

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
        if name in self.elecs:
            self.move_cursors_to_pos()

    def get_devices(self):
        devices = get_devices_names(self.elec_names)


def launch_electrode_gui():
    """Wrapper for the ElcetrodePicker object."""
    app = QApplication(['Electrode Localizer'])
    electrode_gui = ElectrodeGUI()
    electrode_gui.show()
    app.exec_()
'''


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

    def __init__(self, parent=None, width=24, height=16, dpi=300):
        self.fig, self.axes = plt.subplots(2, 3, figsize=(width, height),
                                           dpi=dpi)
        super(SlicePlots, self).__init__(self.fig)


class ElectrodePicker(QMainWindow):
    """Pick electrodes manually using a coregistered MRI and CT."""

    def __init__(self, verbose=True):
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
        """
        # initialize QMainWindow class
        super(ElectrodePicker, self).__init__()

        # change to correct backend
        mpl.use('Qt5Agg')

        # load imaging data
        self.base_path = check_fs_vars()
        self.load_image_data()

        self.elec_radius = int(np.mean(ELEC_PLOT_SIZE) // 100)
        # initialize electrode data
        self.elec_index = 0
        self.elecs = load_electrodes()

        self.elec_names = list(self.elecs.keys())
        for name in load_electrode_names():
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
        if name in self.elecs:
            self.move_cursors_to_pos()

    def load_image_data(self):
        # prepare MRI data
        self.img_data = load_image_data('mri', 'brain.mgz', 'img_pipe.recon',
                                        reorient=True)  # convert to ras
        self.mri_min = self.img_data.min()
        self.mri_max = self.img_data.max()

        # ready ct
        self.ct_data = load_image_data('CT', 'rCT.nii', 'coreg_CT_MR',
                                       reorient=True)  # convert to ras)
        self.ct_min = np.nanmin(self.ct_data)
        self.ct_max = np.nanmax(self.ct_data)

        # prepare pial data
        self.pial_data = dict()
        for hemi in ('lh', 'rh'):
            self.pial_data[hemi] = load_image_data(
                'surf', f'{hemi}.pial.filled.mgz', 'label', reorient=True)
            self.pial_data[hemi] = binary_closing(self.pial_data[hemi])

        # This is the current slice for indexing (as integers for indexing)
        self.current_slice = VOXEL_SIZES // 2

    def make_elec_image(self, axis):
        """Make electrode data higher resolution so it looks better."""
        elec_image = np.zeros(ELEC_PLOT_SIZE) * np.nan
        vx, vy, vz = VOXEL_SIZES

        def color_elec_radius(elec_image, xf, yf, group, radius):
            '''Take the fraction across each dimension of the RAS
               coordinates converted to xyz and put a circle in that
               position in this larger resolution image.'''
            ex, ey = np.round(np.array([xf, yf]) * ELEC_PLOT_SIZE).astype(int)
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if (i**2 + j**2)**0.5 < radius:
                        # negative y because y axis is inverted
                        elec_image[-(ey + i), ex + j] = group
            return elec_image

        for name in self.elecs:
            # move from middle-centered (half coords positive, half negative)
            # to bottom-left corner centered (all coords positive).
            xyz = self.RAS_to_cursors(name)
            # check if closest to that voxel
            dist = np.round(xyz[axis]).astype(int) - self.current_slice[axis]
            if abs(dist) < self.elec_radius:
                x, y, z = xyz
                group = self.elecs[name][3]
                r = self.elec_radius - np.round(abs(dist)).astype(int)
                if axis == 0:
                    elec_image = color_elec_radius(
                        elec_image, y / vy, z / vz, group, r)
                elif axis == 1:
                    elec_image = color_elec_radius(
                        elec_image, x / vx, z / vx, group, r)
                elif axis == 2:
                    elec_image = color_elec_radius(
                        elec_image, x / vx, y / vy, group, r)
        return elec_image

    def make_slice_plots(self):
        self.plt = SlicePlots(self)
        # Plot sagittal (0), coronal (1) or axial (2) view
        self.images = dict(mri=list(), ct=dict(), elec=dict(),
                           pial=dict(lh=list(), rh=list()),
                           cursor=dict(), cursor2=dict())
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
                    self.make_elec_image(axis),
                    aspect='auto', extent=IMG_RANGES[axis],
                    cmap=ELECTRODE_CMAP, alpha=1, vmin=0, vmax=N_COLORS)
                self.images['cursor'][(axis2, axis)] = \
                    self.plt.axes[axis2, axis].plot(
                    (self.current_slice[1], self.current_slice[1]),
                    (0, VOXEL_SIZES[axis]), color=[0, 1, 0], linewidth=0.25)[0]
                self.images['cursor2'][(axis2, axis)] = \
                    self.plt.axes[axis2, axis].plot(
                    (0, VOXEL_SIZES[axis]), color=[0, 1, 0], linewidth=0.25)[0]
                self.plt.axes[axis2, axis].set_facecolor('k')
                # clean up excess plot text, invert
                self.plt.axes[axis2, axis].invert_yaxis()
                self.plt.axes[axis2, axis].set_xticks([])
                self.plt.axes[axis2, axis].set_yticks([])
                # label axes
                self.plt.axes[axis2, axis].set_xlabel(IMG_LABELS[axis][0],
                                                      fontsize=4)
                self.plt.axes[axis2, axis].set_ylabel(IMG_LABELS[axis][1],
                                                      fontsize=4)
                self.plt.axes[axis2, axis].axis(IMG_RANGES[axis])
        self.plt.fig.suptitle('', fontsize=14)
        self.plt.fig.tight_layout()
        self.plt.fig.subplots_adjust(left=0.03, bottom=0.07,
                                     wspace=0.15, hspace=0.15)
        self.plt.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.plt.fig.canvas.mpl_connect('button_release_event', self.on_click)

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
            color = QtGui.QColor()
            color.setRgb(*(255 * np.array(UNIQUE_COLORS[i % N_COLORS])
                           ).round().astype(int))
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

    def set_elec_names(self):
        self.elec_list_model = QtGui.QStandardItemModel(self.elec_list)
        for name in self.elec_names:
            self.elec_list_model.appendRow(QtGui.QStandardItem(name))
        for name in self.elecs:
            self.color_list_item(name=name)
        self.elec_list.setModel(self.elec_list_model)
        self.elec_list.clicked.connect(self.change_elec)
        if self.elec_names or self.elecs:
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
        rgb = (255 * np.array(UNIQUE_COLORS[group % N_COLORS])
               ).round().astype(int)
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
        if name in self.elecs:
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
            group = self.elecs[name][3]
            color.setRgb(*[c * 255 for c in
                           UNIQUE_COLORS[int(group) % N_COLORS]])
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
            self.elecs[name] = \
                self.cursors_to_RAS().tolist() + [self.get_group(), 'n/a']
            self.color_list_item()
            self.update_elec_images(draw=True)
            save_electrodes(self.elecs)
            self.next_elec()

    @pyqtSlot()
    def remove_elec(self):
        name = self.get_current_elec()
        if name in self.elecs:
            self.color_list_item(clear=True)
            self.elecs.pop(name)
            save_electrodes(self.elecs)
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
        return np.array(self.elecs[name][:3]) + VOXEL_SIZES // 2

    def launch_3D_viewer(self):
        """Launch 3D viewer to visualize electrodes."""
        # Get appropriate hemisphere
        plot_brain(get_rois(), opacity=0.3)


def launch_electrode_picker():
    """Wrapper for the ElcetrodePicker object."""
    app = QApplication(['Electrode Picker'])
    electrode_picker = ElectrodePicker()
    electrode_picker.show()
    app.exec_()
