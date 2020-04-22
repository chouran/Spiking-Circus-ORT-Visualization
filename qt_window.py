from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore

#from PyQt5.QtWidgets import*
#from PyQt5.QtCore import*
#from PyQt5.QtGui import*

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

        layout = QtWidgets.QGridLayout()
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

        widget_th = QtWidgets.QCheckBox()
        widget_th.setText('See Thresholds')
        widget_th.setCheckState(QtCore.Qt.Checked)
        widget_th.stateChanged.connect(self.see_th)

        widget_spikes = QtWidgets.QCheckBox()
        widget_spikes.setText('See Spikes')
        widget_spikes.setCheckState(QtCore.Qt.Checked)
        widget_spikes.stateChanged.connect(self.see_spikes)

        """"
        #Scale widgets
        widget_x = QtWidgets.QCheckBox()
        widget_y = QtWidgets.QCheckBox()
        widget_x.setText('x_axis')
        widget_y.setText('y_axis')
        box_button = QtWidgets.QButtonGroup()
        box_button.addButton(widget_x, -1)
        box_button.addButton(widget_y, -1)
        widget_x.setCheckState(QtCore.Qt.Checked)
        widget_x.stateChanged.connect(self.scale)
        #widget_y.stateChanged.connect(self.scale(x,self))
        widget_y.setCheckState(QtCore.Qt.Checked)

        L = [['Up',[0,2]], ['Down',[0,-2]], ['Right', [2,0]], ['Left', [-2,0]]]
        #zoom_list = [['Up', 3], ['Down', 2], ['Right', 1], ['Left', 0]]
        for i in range (4):
            widget_zoom = QtWidgets.QPushButton()
            widget_zoom.setShortcut(QtGui.QKeySequence("Ctrl+"+L[i][0]))
            widget_zoom.setAutoRepeatInterval(200)
            widget_zoom.clicked.connect(lambda i=i: self.zoom(L[i][1]))
            layout.addWidget(widget_zoom, i+4, 0)
            widget_zoom.setFlat(True)
        """

        widget_zoom_in_y = QtWidgets.QPushButton()
        widget_zoom_in_y.setShortcut(QtGui.QKeySequence("Ctrl+Up"))
        widget_zoom_in_y.setAutoRepeatInterval(200)
        widget_zoom_in_y.clicked.connect(lambda : self.zoom([0, 2]))
        layout.addWidget(widget_zoom_in_y, 4, 0)
        widget_zoom_in_y.setFlat(True)

        widget_zoom_out_y = QtWidgets.QPushButton()
        widget_zoom_out_y.setShortcut(QtGui.QKeySequence("Ctrl+Down"))
        widget_zoom_out_y.setAutoRepeatInterval(200)
        widget_zoom_out_y.clicked.connect(lambda : self.zoom([0, -2]))
        layout.addWidget(widget_zoom_out_y, 5, 0)
        widget_zoom_out_y.setFlat(True)

        widget_zoom_in_x = QtWidgets.QPushButton()
        widget_zoom_in_x.setShortcut(QtGui.QKeySequence("Ctrl+Right"))
        widget_zoom_in_x.setAutoRepeatInterval(200)
        widget_zoom_in_x.clicked.connect(lambda : self.zoom([2, 0]))
        layout.addWidget(widget_zoom_in_x, 6, 0)
        widget_zoom_in_x.setFlat(True)

        widget_zoom_out_x = QtWidgets.QPushButton()
        widget_zoom_out_x.setShortcut(QtGui.QKeySequence("Ctrl+Left"))
        widget_zoom_out_x.setAutoRepeatInterval(200)
        widget_zoom_out_x.clicked.connect(lambda : self.zoom([-2, 0]))
        layout.addWidget(widget_zoom_out_x, 7, 0)
        widget_zoom_out_x.setFlat(True)

        # Layout
        layout.addWidget(widget_th, 0, 0)
        layout.addWidget(widget_spikes, 1, 0)
        layout.addWidget(signals_widget, 0,1)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    # Create the slot associated to the toolbar button
    def button_click(self, s):
        print ("T'as cliqué", s)

    #Connect the spin Box value to the thresholds
    def th_value(self, t):
         #print (" Threshold value = ", t)
         self._canvas.update_threshold(t)

    def see_th(self, t):
        self._canvas.see_thresholds(t)

    def see_spikes(self, s):
        print(s)
        self._canvas.see_spikes(s)

    def zoom(self, l):
        print(l)
        self._canvas.zoom(l)





app = QtWidgets.QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()