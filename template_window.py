# coding=utf-8
try:
    # Python 2 compatibility.
    from PyQt4.QtCore import Qt
    from PyQt4.QtGui import QMainWindow, QLabel, QDoubleSpinBox, QSpacerItem, \
        QSizePolicy, QGroupBox, QGridLayout, QLineEdit, QDockWidget, QListWidget, \
        QListWidgetItem, QAbstractItemView, QCheckBox, QTableWidget, QTableWidgetItem
except ImportError:  # i.e. ModuleNotFoundError
    # Python 3 compatibility.
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QMainWindow, QLabel, QDoubleSpinBox, QSpacerItem, \
        QSizePolicy, QGroupBox, QGridLayout, QLineEdit, QDockWidget, QListWidget, \
        QListWidgetItem, QAbstractItemView, QCheckBox, QTableWidget, QTableWidgetItem

import utils.widgets as wid
from views.templates import TemplateCanvas, TemplateControl
from views.electrodes import MEACanvas
from views.rates import RateCanvas, RateControl
from views.isis import ISICanvas

from thread import ThreadORT
from circusort.io.probe import load_probe
from circusort.io.template import load_template_from_dict
import numpy as np

from circusort.obj.cells import Cells
from circusort.obj.cell import Cell
from circusort.obj.train import Train
from circusort.obj.amplitude import Amplitude


class TemplateWindow(QMainWindow, wid.CustomWidget):

    def __init__(self, params_pipe, number_pipe, templates_pipe, spikes_pipe,
                 probe_path=None, screen_resolution=None):

        QMainWindow.__init__(self)

        self.setDockOptions(QMainWindow.AllowTabbedDocks | QMainWindow.AllowNestedDocks | QMainWindow.VerticalTabs)
        self.setDockNestingEnabled(True)

        # Receive parameters.
        params = params_pipe[0].recv()
        self.probe = load_probe(probe_path)
        self._nb_samples = params['nb_samples']
        self._sampling_rate = params['sampling_rate']
        self._display_list = []

        self._params = {
            'nb_samples': self._nb_samples,
            'sampling_rate': self._sampling_rate,
            'time': {
                'min': 10.0,  # ms
                'max': 100.0,  # ms
                'init': 100.0,  # ms
            },
            'voltage': {
                'min': -200,  # µV
                'max': 20e+1,  # µV
                'init': 50.0,  # µV
            },
            'templates': self._display_list
        }

        self.cells = Cells({})
        self.bin_size = 1
        self._nb_buffer = 0

        # TODO ISI
        self.isi_bin_width, self.isi_x_max = 2, 25.0

        # Load the  canvas
        self._canvas_loading(probe_path=probe_path)
        self._control_loading()

        # Load the dock widget
        self._info_dock_widgets(probe_path=probe_path)

        #TODO create a TableWidget method

        self._selection_templates = QTableWidget()
        self._selection_templates.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self._selection_templates.setColumnCount(3)
        self._selection_templates.setVerticalHeaderLabels(['Nb template', 'Channel', 'Amplitude'])
        self._selection_templates.insertRow(0)
        self._selection_templates.setItem(0, 0, QTableWidgetItem('Nb template'))
        self._selection_templates.setItem(0, 1, QTableWidgetItem('Channel'))
        self._selection_templates.setItem(0, 2, QTableWidgetItem('Amplitude'))

        # Create info grid.
        templates_grid = QGridLayout()
        # # Add Channel selection
        # grid.addWidget(label_selection, 3, 0)
        templates_grid.addWidget(self._selection_templates, 0, 1)

        def add_template():
            items = self._selection_templates.selectedItems()
            self._display_list = []
            for i in range(len(items)):
                self._display_list.append(i)
            self._on_templates_changed()

        # self._selection_templates.itemClicked.connect(add_template)

        # Template selection signals
        self._selection_templates.itemSelectionChanged.connect(lambda: self.selected_templates(
            self.nb_templates))

        # self._selection_templates.itemPressed(0, 1).connect(self.sort_template())

        # # Add spacer.
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        templates_grid.addItem(spacer)

        # Create controls group.
        templates_group = QGroupBox()
        templates_group.setLayout(templates_grid)

        # # Create controls dock.
        templates_dock = QDockWidget()
        templates_dock.setWidget(templates_group)
        templates_dock.setWindowTitle("Channels selection")
        self.addDockWidget(Qt.TopDockWidgetArea, templates_dock, Qt.Horizontal)

        # Create thread.
        thread2 = ThreadORT(number_pipe, templates_pipe, spikes_pipe)
        thread2.number_signal.connect(self._number_callback)
        thread2.reception_signal.connect(self._reception_callback)
        #thread2.start()

        #self.setCentralWidget(QLineEdit())

        # Set window size.
        if screen_resolution is not None:
            screen_width = screen_resolution.width()
            screen_height = screen_resolution.height()
            self.resize(screen_width, screen_height)

        # Set window title.
        self.setWindowTitle("SpyKING Circus ORT - Read 'n' Qt display")


    @property
    def nb_templates(self):
        return len(self.cells)

    # -----------------------------------------------------------------------------
    # Canvas & Control handling
    # -----------------------------------------------------------------------------

    def _canvas_loading(self, probe_path):
        """ Load the vispy canvas from the files """

        self._canvas_mea = MEACanvas(probe_path=probe_path, params=self._params)
        self._canvas_template = TemplateCanvas(probe_path=probe_path, params=self._params)
        self._canvas_rate = RateCanvas(probe_path=probe_path, params=self._params)
        self._canvas_isi = ISICanvas(probe_path=probe_path, params=self._params)


        self.all_canvas = [self._canvas_mea, self._canvas_template, self._canvas_rate, self._canvas_isi]

        """ Transform the vispy canvas into QT canvas """
        self.addDockWidget(Qt.LeftDockWidgetArea, wid.dock_canvas(self._canvas_template))
        self.addDockWidget(Qt.RightDockWidgetArea, wid.dock_canvas(self._canvas_mea))
        self.addDockWidget(Qt.LeftDockWidgetArea, wid.dock_canvas(self._canvas_rate))
        self.addDockWidget(Qt.RightDockWidgetArea, wid.dock_canvas(self._canvas_isi))

    def _control_loading(self):
        """ """
        self.template_control = TemplateControl(self._canvas_template, self._params)
        self.rate_control = RateControl(self._canvas_rate, self.bin_size)

        self.addDockWidget(Qt.TopDockWidgetArea, self.template_control.dock_widget, Qt.Horizontal)
        self.addDockWidget(Qt.TopDockWidgetArea, self.rate_control.dock_widget, Qt.Horizontal)

    def _info_dock_widgets(self, probe_path):
        """ Add the info dock to the GUI"""
        #self._info_time, self._info_buffer, self._info_probe = wid.info_widgets(probe_path=probe_path)

        self._info_time = self.line_edit(label='Time', init_value='0', read_only=True, label_unit='s')
        self._info_buffer = self.line_edit(label='Buffer', init_value='0', read_only=True, label_unit=None)
        self._info_probe = self.line_edit(label='Probe', init_value="{}".format(probe_path),
                                               read_only=True, label_unit=None)

        self._info_dock = wid.dock_control('Left', 'Info', self._info_time,
                                           self._info_buffer, self._info_probe)
        self.addDockWidget(Qt.TopDockWidgetArea, self._info_dock, Qt.Horizontal)

    #def _template_table(self):

    # -----------------------------------------------------------------------------
    # Data handling
    # -----------------------------------------------------------------------------

    def _number_callback(self, number):

        self._nb_buffer = float(number)
        nb_buffer = u"{}".format(number)
        #self._info_buffer['widget'].setText(nb_buffer)

        txt_time = u"{:8.3f}".format(self._nb_buffer * float(self._nb_samples) / self._sampling_rate)
        #self._info_time['widget'].setText(txt_time)

        return

    def _reception_callback(self, templates, spikes):
        bar = None
        if templates is not None:
            bar = []
            for i in range(len(templates)):
                mask = spikes['templates'] == i
                template = load_template_from_dict(templates[i], self.probe)

                new_cell = Cell(template, Train([]), Amplitude([], []))
                self.cells.append(new_cell)
                self._selection_templates.insertRow(self.nb_templates)

                bar += [template.center_of_mass(self.probe)]
                channel = template.channel
                amplitude = template.peak_amplitude()
                # self._selection_templates.setItem(self.nb_templates, 0, QTableWidgetItem("Template %d" %self.nb_templates))
                # self._selection_templates.setItem(self.nb_templates, 1, QTableWidgetItem(str(bar)))
                self._selection_templates.setItem(self.nb_templates, 0, QTableWidgetItem(str(self.nb_templates)))
                self._selection_templates.setItem(self.nb_templates, 1, QTableWidgetItem(str(channel)))
                self._selection_templates.setItem(self.nb_templates, 2, QTableWidgetItem(str(amplitude)))
                # item = QListWidgetItem("Template %i" % self.nb_templates)
                # self._selection_templates.addItem(item)
                # self._selection_templates.item(i).setSelected(False)
                # self.nb_templates += 1
                # print(bar.shape, bar)

        if spikes is not None:
            self.cells.add_spikes(spikes['spike_times'], spikes['amplitudes'], spikes['templates'])
            self.cells.set_t_max(self._nb_samples * self._nb_buffer / self._sampling_rate)
            to_display = self.cells.rate(self.bin_size)


        # for canvas in self.all_canvas:
        #     data_requested = canvas.data_requested
        #     to_send = {}
        #     for 
        #     canvas.on_reception

        self._canvas_template.on_reception(templates, self.nb_templates)
        self._canvas_mea.on_reception(bar, self.nb_templates)
        # TODO Cells rate
        self._canvas_rate.on_reception(self.cells.rate(self.bin_size))

        # TODO : ISI If we want to display the ISI also
        # isi = self.cells.interspike_interval_histogram(self.isi_bin_width, self.isi_x_max=25.0)
        isi = self.cells.interspike_interval_histogram(self.isi_bin_width, self.isi_x_max)
        self._canvas_isi.on_reception(isi)

        return

    def selected_templates(self, max_templates):
        list_templates = []
        list_channels = []
        for i in range(max_templates + 1):
            if i != 0 and \
                    self._selection_templates.item(i, 0).isSelected() and \
                    self._selection_templates.item(i, 1).isSelected() and \
                    self._selection_templates.item(i, 2).isSelected():
                list_templates.append(i - 1)
                list_channels.append(int(self._selection_templates.item(i, 1).text()))

        for canvas in self.all_canvas:
            self._canvas_template.highlight_selection(list_templates)
    
        return

