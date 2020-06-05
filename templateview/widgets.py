# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Create control widgets
# -----------------------------------------------------------------------------


class ControlWidget:
    def __init__(self):

    def double_spin_box(self, **kwargs):

        """""
        kwargs parameters 
        label : str
        unit : str
        min value : float
        max value : float
        step : float
        init_value : float
        
        return a dictionnary with the following objects : label, double spin box, unit_label
        
        """""

        dsb_widget = {}
        dsb = QDoubleSpinBox()
        if 'label' in kwargs.keys():
            label_dsb = QLabel()
            label_dsb.setText(kwargs['label'])
            dsb_widget['label'] = label_dsb
        if 'min_value' in kwargs.keys():
            dsb.setMinimum(kwargs['min_value'])
        if 'max_value' in kwargs.keys():
            dsb.setMaximum(kwargs['max_value'])
        if 'step' in kwargs.keys():
            dsb.setSingleStep(kwargs['step'])

        dsb_widget['widget'] = dsb_widget

        if 'unit' in kwargs.keys():
            label_unit = QLabel()
            label_unit.setText(kwargs['unit'])
            dsb_widget['unit'] = label_unit

        return dsb_widget

    def checkbox(self, **kwargs):

        """""
        kwargs param
        label : str
        init_state : bool
        
        return a dictionnary with the following objects : label, checkbox
        """"

        cb_widget = {}
        cb = QCheckBox()

        if 'label' in kwargs.keys():
            label_cb = QLabel()
            label_cb.setText(kwargs['label'])
            cb_widget['label'] = label_cb
        if 'init_state' in kwargs.keys():
            cb.setChecked(kwargs['init_state'])
        cb_widget['widget'] = cb

        return cb_widget

    def grid_layout(self, *args):

        """"
        args : dict of widgets
        return a grid layout object with the widgets correctly  positioned
        """

        grid_layout = QGridLayout()
        for widget_dict in args:
            i = 0  # line_number
            for name, widget_obj in widget_dict.items():
                j = 0  # column number
                grid_layout.addWidget(widget_obj, i, j)
                j += 1
            i += 1

        return grid_layout


# -----------------------------------------------------------------------------
# Create layout
# -----------------------------------------------------------------------------





