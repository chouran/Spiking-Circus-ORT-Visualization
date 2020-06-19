import numpy as np
from multiprocessing import Pipe
from gui_process import GUIProcess
from circusort.io.probe import load_probe

class ORTSimulator(object):
    """Peak displayer"""

    def __init__(self, **kwargs):
        """Initialization"""

        self.nb_samples = 1024
        self.dtype = 'float32'
        self.sampling_rate = 20000
        self.probe_path = 'probe.prb'
        self.probe = load_probe(self.probe_path)
        self.nb_channels = self.probe.nb_channels
        self.export_peaks = True

        self._params_pipe = Pipe()
        self._number_pipe = Pipe()
        self._data_pipe = Pipe()
        self._mads_pipe = Pipe()
        self._peaks_pipe = Pipe()
        self._qt_process = GUIProcess(self._params_pipe, self._number_pipe, self._data_pipe, self._mads_pipe,
                                      self._peaks_pipe, probe_path=self.probe_path)

        self._is_mad_reception_blocking = False
        self._is_peak_reception_blocking = False
        self._qt_process.start()
        self.number = 0

        self._params_pipe[1].send({
            'nb_samples': self.nb_samples,
            'sampling_rate': self.sampling_rate,
        })

        return

    def run(self):

        while True:
            # Here we are increasing the counter
            self.number += 1

            # Here we need to generate the fake data
            data = np.random.randn(self.nb_samples, self.nb_channels).astype(np.float32)
            # Here we are generating fake thresholds
            mads = np.std(data, 0)

            self._number_pipe[1].send(self.number)
            self._data_pipe[1].send(data)
            self._mads_pipe[1].send(mads)

            if self.export_peaks:
                peaks = {}
                for i in range(self.nb_channels):
                    peaks[i] = np.where(data[i] > mads[i])[0]
            else:
                peaks = None
            self._peaks_pipe[1].send(peaks)


if __name__ == "__main__":
    # execute only if run as a script
    simulator = ORTSimulator()
    simulator.run()
