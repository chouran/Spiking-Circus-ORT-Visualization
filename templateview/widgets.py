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

    def double_spin_box(self, label=None, unit=None,
                        min_value=None, max_value=None, step=None, value=None):
        label_dsb = QLabel()
        label_dsb.setText(label)
        label_unit = QLabel()
        label_unit.setText(unit)

        dsb = QDoubleSpinBox()
        dsb.setMinimum(min_value)
        dsb.setMaximum(max_value)
        dsb.setSingleStep(step)
        dsb.setValue(value)

        dsb_widget = {"label" : label_dsb, "widget" : dsb, "unit" : label_unit}
        return (dsb_widget)

    def checkbox(self, label=None, init_value=None):

        label_cb = QLabel()
        label_cb.setText(label)
        cb = QCheckBox()
        cb.setChecked(init_value)

        cb_widget = {"label" : label_cb, "widget" : cb}
        return(cb_widget)


# -----------------------------------------------------------------------------
# Create layout
# -----------------------------------------------------------------------------

class GridLayout:
    def __init__(self, *args, **kwargs):
        self.grid_layout = QGridLayout()
        for widget_dict in args:
            i=0                        #line_number
            for name, widget_obj in widget_dict.items():
                j = 0                  #column number
                self.grid_layout.addWidget(widget_obj, i, j)
                j+=1
            i+=1




