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

from template_canvas import TemplateCanvas
from electrode_canvas import MEACanvas
from thread import Thread
from circusort.io.probe import load_probe
from circusort.io.template import load_template_from_dict
import numpy as np

from circusort.obj.cells import Cells
from circusort.obj.cell import Cell
from circusort.obj.train import Train
from circusort.obj.amplitude import Amplitude

class TemplateWindow(QMainWindow):

    def __init__(self, params_pipe, number_pipe, templates_pipe, spikes_pipe,
                 probe_path=None, screen_resolution=None):

        QMainWindow.__init__(self)

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
                'init': 10.0,  # µV
            },
            'mads': {
                'min': 0.0,  # µV
                'max': 100,  # µV
                'init': 3,  # µV
            },
            'templates': self._display_list
        }

        self._canvas_mea = MEACanvas(probe_path=probe_path, params=self._params)
        self._canvas = TemplateCanvas(probe_path=probe_path, params=self._params)

        self.cells = Cells({})
        self._nb_buffer = 0
        
        canvas_template_widget = self._canvas.native
        canvas_mea = self._canvas_mea.native

        # Create controls widgets.
        label_time = QLabel()
        label_time.setText(u"time")
        label_time_unit = QLabel()
        label_time_unit.setText(u"ms")

        self._dsp_time = QDoubleSpinBox()
        self._dsp_time.setMinimum(self._params['time']['min'])
        self._dsp_time.setMaximum(self._params['time']['max'])
        self._dsp_time.setValue(self._params['time']['init'])
        self._dsp_time.valueChanged.connect(self._on_time_changed)

        label_voltage = QLabel()
        label_voltage.setText(u"voltage")
        label_voltage_unit = QLabel()
        label_voltage_unit.setText(u"µV")
        self._dsp_voltage = QDoubleSpinBox()
        self._dsp_voltage.setMinimum(self._params['voltage']['min'])
        self._dsp_voltage.setMaximum(self._params['voltage']['max'])
        self._dsp_voltage.setValue(self._params['voltage']['init'])
        self._dsp_voltage.valueChanged.connect(self._on_voltage_changed)
       
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

        
        #self._selection_channels.setGeometry(QtCore.QRect(10, 10, 211, 291))
        # for i in range(self.nb_templates):
        #     numRows = self.tableWidget.rowCount()
        #     self.tableWidget.insertRow(numRows)

        #     item = QTableWidgetItem("Template %i" % i)
        #     self._selection_templates.addItem(item)
        #     self._selection_templates.item(i).setSelected(False)

        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        # Create controls grid.
        grid = QGridLayout()
        # # Add time row.
        grid.addWidget(label_time, 0, 0)
        grid.addWidget(self._dsp_time, 0, 1)
        grid.addWidget(label_time_unit, 0, 2)
        # # Add voltage row.
        grid.addWidget(label_voltage, 1, 0)
        grid.addWidget(self._dsp_voltage, 1, 1)
        grid.addWidget(label_voltage_unit, 1, 2)

        # # Add spacer.
        grid.addItem(spacer)

        # # Create info group.
        controls_group = QGroupBox()
        controls_group.setLayout(grid)

        # Create info grid.
        templates_grid = QGridLayout()
        # # Add Channel selection
        #grid.addWidget(label_selection, 3, 0)
        templates_grid.addWidget(self._selection_templates, 0, 1)

        def add_template():
            items = self._selection_templates.selectedItems()
            self._display_list = []
            for i in range(len(items)):
                self._display_list.append(i)
            self._on_templates_changed()

        #self._selection_templates.itemClicked.connect(add_template)

        # Template selection signals
        self._selection_templates.itemSelectionChanged.connect(lambda: self.selected_templates(
            self.nb_templates))
        #self._selection_templates.itemPressed(0, 1).connect(self.sort_template())



        # # Add spacer.
        templates_grid.addItem(spacer)

        # Create controls group.
        templates_group = QGroupBox()
        templates_group.setLayout(templates_grid)

        # # Create controls dock.
        templates_dock = QDockWidget()
        templates_dock.setWidget(templates_group)
        templates_dock.setWindowTitle("Channels selection")

        # # Create controls dock.
        control_dock = QDockWidget()
        control_dock.setWidget(controls_group)
        control_dock.setWindowTitle("Controls")

        # Create info widgets.
        label_time = QLabel()
        label_time.setText(u"time")
        self._label_time_value = QLineEdit()
        self._label_time_value.setText(u"0")
        self._label_time_value.setReadOnly(True)
        self._label_time_value.setAlignment(Qt.AlignRight)
        label_time_unit = QLabel()
        label_time_unit.setText(u"s")
        info_buffer_label = QLabel()
        info_buffer_label.setText(u"buffer")
        self._info_buffer_value_label = QLineEdit()
        self._info_buffer_value_label.setText(u"0")
        self._info_buffer_value_label.setReadOnly(True)
        self._info_buffer_value_label.setAlignment(Qt.AlignRight)
        info_buffer_unit_label = QLabel()
        info_buffer_unit_label.setText(u"")
        info_probe_label = QLabel()
        info_probe_label.setText(u"probe")
        info_probe_value_label = QLineEdit()
        info_probe_value_label.setText(u"{}".format(probe_path))
        info_probe_value_label.setReadOnly(True)
        # TODO place the following info in another grid?
        info_probe_unit_label = QLabel()
        info_probe_unit_label.setText(u"")

        info_spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        # Create info grid.
        info_grid = QGridLayout()
        # # Time row.
        info_grid.addWidget(label_time, 0, 0)
        info_grid.addWidget(self._label_time_value, 0, 1)
        info_grid.addWidget(label_time_unit, 0, 2)
        # # Buffer row.
        info_grid.addWidget(info_buffer_label, 1, 0)
        info_grid.addWidget(self._info_buffer_value_label, 1, 1)
        info_grid.addWidget(info_buffer_unit_label, 1, 2)
        # # Probe row.
        info_grid.addWidget(info_probe_label, 2, 0)
        info_grid.addWidget(info_probe_value_label, 2, 1)
        info_grid.addWidget(info_probe_unit_label, 2, 2)
        # # Spacer.
        info_grid.addItem(info_spacer)

        # Create info group.
        info_group = QGroupBox()
        info_group.setLayout(info_grid)

        # Create info dock.
        info_dock = QDockWidget()
        info_dock.setWidget(info_group)
        info_dock.setWindowTitle("Info")

        # Create thread.
        thread = Thread(number_pipe, templates_pipe, spikes_pipe)
        thread.number_signal.connect(self._number_callback)
        thread.reception_signal.connect(self._reception_callback)
        thread.start()

        # Add dockable windows.
        self.addDockWidget(Qt.LeftDockWidgetArea, control_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, info_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, templates_dock)

        # Add Grid Layout for canvas
        canvas_grid = QGridLayout()
        canvas_grid.addWidget(canvas_template_widget, 0, 0)
        canvas_grid.addWidget(canvas_mea, 0, 1)
        canvas_group = QGroupBox()
        canvas_group.setLayout(canvas_grid)

        # Set central widget.
        self.setCentralWidget(canvas_group)
        # Set window size.
        if screen_resolution is not None:
            screen_width = screen_resolution.width()
            screen_height = screen_resolution.height()
            self.resize(screen_width, screen_height)
        # Set window title.
        self.setWindowTitle("SpyKING Circus ORT - Read 'n' Qt display")

        print(" ")  # TODO remove?


    @property
    def nb_templates(self):
        return len(self.cells)

    def _number_callback(self, number):

        self._nb_buffer = float(number)
        text = u"{}".format(number)
        self._info_buffer_value_label.setText(text)

        text = u"{:8.3f}".format(self._nb_buffer * float(self._nb_samples) / self._sampling_rate)
        self._label_time_value.setText(text)

        return

    def _reception_callback(self, templates, spikes):

        for i in range(len(templates)):

            mask = spikes['templates'] == i
            template = load_template_from_dict(templates[i], self.probe) 

            new_cell = Cell(template, Train([]), Amplitude([], []))
            self.cells.append(new_cell)
            self._selection_templates.insertRow(self.nb_templates)

            bar = template.center_of_mass(self.probe)
            channel = template.channel
            amplitude = template.peak_amplitude()

            self._selection_templates.setItem(self.nb_templates, 0, QTableWidgetItem(str(self.nb_templates)))
            self._selection_templates.setItem(self.nb_templates, 1, QTableWidgetItem(str(channel)))
            self._selection_templates.setItem(self.nb_templates, 2, QTableWidgetItem(str(amplitude)))

        print(len(self.cells))

        if spikes is not None:
            self.cells.add_spikes(spikes['spike_times'], spikes['amplitudes'], spikes['templates'])
            self.cells.set_t_max(self._nb_samples*self._nb_buffer/self._sampling_rate)
            #print(self.cells.rate(1))

        self._canvas.on_reception(templates, spikes, self.nb_templates)

        return

    def _on_time_changed(self):

        time = self._dsp_time.value()
        self._canvas.set_time(time)

        return

    def _on_voltage_changed(self):

        voltage = self._dsp_voltage.value()
        self._canvas.set_voltage(voltage)

        return

    def _on_mads_changed(self):

        mads = self._dsp_mads.value()
        self._canvas.set_mads(mads)

        return

    def _on_mads_display(self):

        value = self._display_mads.isChecked()
        self._canvas.show_mads(value)

        return

    def _on_peaks_display(self):

        value = self._display_peaks.isChecked()
        self._canvas.show_peaks(value)

        return

    def _on_templates_changed(self):
        self._canvas.set_templates(self._display_list)

        return

    def selected_templates(self, max_templates):
        # print(self._selection_channels.selectedItems())
        list_templates = []
        list_channels = []
        for i in range(max_templates+1):
            if i != 0 and \
                    self._selection_templates.item(i, 0).isSelected() and \
                    self._selection_templates.item(i, 1).isSelected() and \
                    self._selection_templates.item(i, 2).isSelected():
                list_templates.append(i)
                list_channels.append(int(self._selection_templates.item(i, 1).text()))
        print(list_templates)
        print(list_channels, set(list_channels))
        self._canvas.selected_templates(list_templates)
        self._canvas_mea.selected_channels(list_channels)
        return

    def sort_template(self):
        print('test')
        return

