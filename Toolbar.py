from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import sys

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("Test_toolbar")

        label = QLabel("script test toolbars")
        # Text displayed on top
        label.setAlignment(Qt.AlignHCenter)
        self.setCentralWidget(label)

        toolbar = QToolBar('main toolbar')
        toolbar.setIconSize(QSize(64, 64))
        self.addToolBar(toolbar)

        button_action = QAction(QIcon("brain.png"),'button 1', self) # Button creation
        button_action.setStatusTip("brain button")                   # Informative text
        button_action.triggered.connect(self.button_click)           # Signal slot connection
        button_action.setCheckable(True)    #1
        toolbar.addAction(button_action)

        toolbar.addSeparator()

        button_action2 = QAction(QIcon("android.png"), 'button2', self)
        button_action2.setStatusTip("brain button")
        button_action2.triggered.connect(self.button_click)
        button_action2.setCheckable(True)
        toolbar.addAction(button_action2)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Spiking Circus"))

        toolbar.addSeparator()

        toolbar.addWidget(QCheckBox())

        self.setStatusBar(QStatusBar(self)) #2

        # Menus => pretty easy to manipulate 
        menu = self.menuBar()
        menu.setNativeMenuBar(False) # Disables the native menu bar on Mac. Keep it in mind

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



    # Create the slot associated to the toolbar button
    # s represents its state. Hwv the button is only clickable,
    # not checkable. To associate a state we add #1 & #2
    def button_click(self, s):
        print ("T'as cliqué", s)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()