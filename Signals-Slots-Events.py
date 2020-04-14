from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import sys

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("Buttons")

        layout = QHBoxLayout()

        # Create several buttons with digits
        # the .connect links the signal (aka event) "button pressed" to
        # the slot (aka action) "custom_fn"
        for i in range(10):
            btn = QPushButton(str(i))
            btn.pressed.connect(lambda i=i: self.custom_fn(i))
            layout.addWidget(btn)

        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)

    # SLOT: This has default parameters and can be called without a value
    def custom_fn(self, x):
        print(x)

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()