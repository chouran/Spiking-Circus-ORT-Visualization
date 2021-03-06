# coding=utf-8
try:
    # Python 2 compatibility.
    from PyQt4.QtCore import Qt
    from PyQt4.QtGui import QMainWindow, QLabel, QDoubleSpinBox, QSpacerItem, \
        QSizePolicy, QGroupBox, QGridLayout, QLineEdit, QDockWidget, QListWidget, \
        QListWidgetItem, QAbstractItemView, QCheckBox
except ImportError:  # i.e. ModuleNotFoundError
    # Python 3 compatibility.
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QMainWindow, QLabel, QDoubleSpinBox, QSpacerItem, \
        QSizePolicy, QGroupBox, QGridLayout, QLineEdit, QDockWidget, QListWidget, \
        QListWidgetItem, QAbstractItemView, QCheckBox

from trace_canvas import TraceCanvas
from thread import Thread
from circusort.io.probe import load_probe


class TraceWindow(QMainWindow):

    def __init__(self, params_pipe, number_pipe, data_pipe, mads_pipe, peaks_pipe,
                 probe_path=None, screen_resolution=None):

        QMainWindow.__init__(self)

        # Receive parameters.
        params = params_pipe[0].recv()
        self.probe = load_probe(probe_path)
        self._nb_samples = params['nb_samples']
        self._sampling_rate = params['sampling_rate']
        self._display_list = list(range(self.probe.nb_channels))

        self._params = {
            'nb_samples': self._nb_samples,
            'sampling_rate': self._sampling_rate,
            'time': {
                'min': 10.0,  # ms
                'max': 1000.0,  # ms
                'init': 100.0,  # ms
            },
            'voltage': {
                'min': 10.0,  # µV
                'max': 10e+3,  # µV
                'init': 20.0,  # µV
            },
            'mads': {
                'min': 0.0,  # µV
                'max': 100,  # µV
                'init': 3,  # µV
            },
            'channels': self._display_list
        }

        self._canvas = TraceCanvas(probe_path=probe_path, params=self._params)

        central_widget = self._canvas.native

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

        label_display_mads = QLabel()
        label_display_mads.setText(u"Display Mads")
        self._display_mads = QCheckBox()
        self._display_mads.stateChanged.connect(self._on_mads_display)

        label_display_peaks = QLabel()
        label_display_peaks.setText(u"Display Peaks")
        self._display_peaks = QCheckBox()
        self._display_peaks.stateChanged.connect(self._on_peaks_display)

        label_mads = QLabel()
        label_mads.setText(u"Mads")
        label_mads_unit = QLabel()
        label_mads_unit.setText(u"unit")
        self._dsp_mads = QDoubleSpinBox()
        self._dsp_mads.setMinimum(self._params['mads']['min'])
        self._dsp_mads.setMaximum(self._params['mads']['max'])
        self._dsp_mads.setValue(self._params['mads']['init'])
        self._dsp_mads.valueChanged.connect(self._on_mads_changed)

        label_voltage = QLabel()
        label_voltage.setText(u"voltage")
        label_voltage_unit = QLabel()
        label_voltage_unit.setText(u"µV")
        self._dsp_voltage = QDoubleSpinBox()
        self._dsp_voltage.setMinimum(self._params['voltage']['min'])
        self._dsp_voltage.setMaximum(self._params['voltage']['max'])
        self._dsp_voltage.setValue(self._params['voltage']['init'])
        self._dsp_voltage.valueChanged.connect(self._on_voltage_changed)

        # Color spikes
        self._color_spikes = QCheckBox()
        self._color_spikes.setText('See Spikes color')
        self._color_spikes.setCheckState(Qt.Checked)
        self._color_spikes.stateChanged.connect(self.display_spikes_color)



        # self._selection_channels.setGeometry(QtCore.QRect(10, 10, 211, 291))

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
        # # Add Mads widgets

        grid.addWidget(label_display_mads, 3, 0)
        grid.addWidget(self._display_mads, 3, 1)

        grid.addWidget(label_mads, 4, 0)
        grid.addWidget(self._dsp_mads, 4, 1)
        grid.addWidget(label_mads_unit, 4, 2)

        grid.addWidget(self._color_spikes, 5, 0)

        # # Add spacer.
        grid.addItem(spacer)

        # # Create info group.
        controls_group = QGroupBox()
        controls_group.setLayout(grid)


        self._selection_channels = QListWidget()
        self._selection_channels.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )

        for i in range(self.probe.nb_channels):
            item = QListWidgetItem("Channel %i" % i)
            self._selection_channels.addItem(item)
            self._selection_channels.item(i).setSelected(True)

        def add_channel():
            items = self._selection_channels.selectedItems()
            self._display_list = []
            for i in range(len(items)):
                self._display_list.append(i)
            self._on_channels_changed()

        # self._selection_channels.itemClicked.connect(add_channel)

        nb_channel = self.probe.nb_channels
        self._selection_channels.itemSelectionChanged.connect(lambda: self.selected_channels(nb_channel))

        # Create info grid.
        channels_grid = QGridLayout()
        # # Add Channel selection
        # grid.addWidget(label_selection, 3, 0)
        channels_grid.addWidget(self._selection_channels, 0, 1)

        # # Add spacer.
        channels_grid.addItem(spacer)

        # Create controls group.
        channels_group = QGroupBox()
        channels_group.setLayout(channels_grid)

        # # Create controls dock.
        channels_dock = QDockWidget()
        channels_dock.setWidget(channels_group)
        channels_dock.setWindowTitle("Channels selection")

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
        thread = Thread(number_pipe, data_pipe, mads_pipe, peaks_pipe)
        thread.number_signal.connect(self._number_callback)
        thread.reception_signal.connect(self._reception_callback)
        thread.start()

        # Add dockable windows.
        self.addDockWidget(Qt.LeftDockWidgetArea, control_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, info_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, channels_dock)
        # Set central widget.
        self.setCentralWidget(central_widget)
        # Set window size.
        if screen_resolution is not None:
            screen_width = screen_resolution.width()
            screen_height = screen_resolution.height()
            self.resize(screen_width, screen_height)
        # Set window title.
        self.setWindowTitle("SpyKING Circus ORT - Read 'n' Qt display")

        print(" ")  # TODO remove?

    def _number_callback(self, number):

        text = u"{}".format(number)
        self._info_buffer_value_label.setText(text)

        text = u"{:8.3f}".format(float(number) * float(self._nb_samples) / self._sampling_rate)
        self._label_time_value.setText(text)

        return

    def _reception_callback(self, data, mads, peaks):

        self._canvas.on_reception(data, mads, peaks)

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

    def _on_channels_changed(self):
        self._canvas.set_channels(self._display_list)

        return

    def display_spikes_color(self, s):
        self._canvas.color_spikes(s)

    def selected_channels(self, max_channel):
        # print(self._selection_channels.selectedItems())
        list_channel = []
        for i in range(max_channel):
            if self._selection_channels.item(i).isSelected():
                list_channel.append(i)
        self._canvas.selected_channels(list_channel)
        return
