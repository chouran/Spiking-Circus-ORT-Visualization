from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore

from PyQt5.QtWidgets import*
from PyQt5.QtCore import*
from PyQt5.QtGui import*

import sys
from rt_signals import *

class Color(QtWidgets.QWidget):

    def __init__(self, color, *args, **kwargs):
        super(Color, self).__init__(*args, **kwargs)
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Import Vispy Canvas
        self._canvas = SignalCanvas()
        self._canvas.native.setParent(self)
        signals_widget = self._canvas.native

        #Window Custom
        self.setWindowTitle("Real time signals")

        toolbar = QtWidgets.QToolBar('main toolbar')
        toolbar.setIconSize(QtCore.QSize(32, 32))
        self.addToolBar(toolbar)

        button_action = QtWidgets.QAction(QtGui.QIcon("brain.png"),
                        'button 1', self)                            # Button creation
        button_action.setStatusTip("brain button")                   # Informative text
        button_action.triggered.connect(self.button_click)           # Signal slot connection
        button_action.setCheckable(True)    #1
        button_action.setShortcut(QtGui.QKeySequence("Ctrl+p"))            # Shortcut
        toolbar.addAction(button_action)

        toolbar.addSeparator()

        button_action2 = QtWidgets.QAction(QtGui.QIcon("android.png"), 'button2', self)
        button_action2.setStatusTip("android button")
        button_action2.triggered.connect(self.button_click)
        button_action2.setCheckable(True)
        toolbar.addAction(button_action2)

        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel("Spiking Circus"))
        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QCheckBox())
        self.setStatusBar(QtWidgets.QStatusBar(self))

        # Menus => pretty easy to manipulate
        menu = self.menuBar()

        file_menu = menu.addMenu("File")
        file_menu.addAction(button_action)
        file_menu.addSeparator()
        file_menu.addAction(button_action2)

        file_submenu = file_menu.addAction(button_action2)
        menu.addSeparator()
        edit_menu = menu.addMenu("Edit")
        menu.addSeparator()
        options_menu = menu.addMenu("Options")
        menu.addSeparator()
        help_menu = menu.addMenu("Help")

        # Threshold widget
        widget_seuil = QDoubleSpinBox()
        widget_seuil.setMinimum(-1.0)
        widget_seuil.setMaximum(+1.0)
        widget_seuil.setSuffix(" V")
        widget_seuil.setSingleStep(0.01)
        widget_seuil.valueChanged.connect(self.th_value)

        # Layout
        layout = QGridLayout()
        layout.addWidget(widget_seuil, 0, 0)
        layout.addWidget(signals_widget, 0,1)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    # Create the slot associated to the toolbar button
    def button_click(self, s):
        print ("T'as cliqué", s)

    #Connect the spin Box value to the thresholds
    def th_value(self, t):
         #print (" Threshold value = ", t)
         self._canvas.update_threshold(t)

app = QtWidgets.QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()