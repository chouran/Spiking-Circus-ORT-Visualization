from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import sys

# Subclass QMainWindow to customise the application's main window
# Inherit the QMainWindow attributes
class mainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        # Important or it won't work
        super(mainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("Simple Window")

        # Text to display in the widget
        label = QLabel("BOUH")
        # Alignment inside the widget ! not the window
        label.setAlignment(Qt.AlignCenter)

        # Set the central widget of the Window.
        # Widget will expand to take up all the space in the window by default.
        self.setCentralWidget(label)

#Start the event loop
app = QApplication(sys.argv)

window = mainWindow()

#By default the window is hidden
window.show()
app.exec_()
