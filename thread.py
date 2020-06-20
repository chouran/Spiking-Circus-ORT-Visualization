try:
    from PyQt4.QtCore import QThread, pyqtSignal  # Python 2 compatibility.
except ImportError:  # i.e. ModuleNotFoundError
    from PyQt5.QtCore import QThread, pyqtSignal  # Python 3 compatibility.


all_pipes = ['number', 'templates', 'peaks', 'spikes', 'raw_data', 'thresholds']

class ThreadORT(QThread):

    number_signal = pyqtSignal(object)
    reception_signal = pyqtSignal(object, object)

    def __init__(self, all_pipes, sleep_duration=90):

        QThread.__init__(self)

        self.pipes = {}
        self.sleep_duration = sleep_duration

        for key, value in all_pipes.items():
            self.pipes[key] = value

    def __del__(self):

        self.wait()

    def run(self):

        while True:

            to_send = {}
            for key, value in self.pipes.items():

                to_send[key] = value[0].recv()

            # Emit signal.
            self.reception_signal.emit(to_send)
            # Sleep.
            self.msleep(self.sleep_duration)  # TODO compute this duration (sampling rate & number of samples per buffer).

        return
