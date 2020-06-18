try:
    from PyQt4.QtCore import QThread, pyqtSignal  # Python 2 compatibility.
except ImportError:  # i.e. ModuleNotFoundError
    from PyQt5.QtCore import QThread, pyqtSignal  # Python 3 compatibility.


class ThreadORT(QThread):

    number_signal = pyqtSignal(object)
    reception_signal = pyqtSignal(object, object)

    def __init__(self, number_pipe, templates_pipe, spikes_pipe):

        QThread.__init__(self)

        self._number_pipe = number_pipe
        self._templates_pipe = templates_pipe
        self._spikes_pipe = spikes_pipe

    def __del__(self):

        self.wait()

    def run(self):

        while True:

            # Process number.
            number = self._number_pipe[0].recv()
            self.number_signal.emit(str(number))
            # Process templates.
            templates = self._templates_pipe[0].recv()
            # Process spikes.
            spikes = self._spikes_pipe[0].recv()
            # Emit signal.
            self.reception_signal.emit(templates, spikes)
            # Sleep.
            self.msleep(90)  # TODO compute this duration (sampling rate & number of samples per buffer).

        return
