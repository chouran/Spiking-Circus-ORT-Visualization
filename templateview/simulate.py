import numpy as np
from multiprocessing import Pipe
from gui_process import GUIProcess
from circusort.io.probe import load_probe
from circusort.io.template_store import load_template_store

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
        self.templates = load_template_store('data/templates.h5', 'probe.prb')

        self._params_pipe = Pipe()
        self._number_pipe = Pipe()
        self._templates_pipe = Pipe()
        self._spikes_pipe = Pipe()
        self._qt_process = GUIProcess(self._params_pipe, self._number_pipe, self._templates_pipe, self._spikes_pipe,
                                      probe_path=self.probe_path)

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
            index = 0

            if self.number == self.templates[index].creation_time:
                templates = self.templates[index].todict()
            else:
                templates = None
            self._number_pipe[1].send(self.number)
            self._templates_pipe[1].send(templates)
            self._spikes_pipe[1].send(None)

if __name__ == "__main__":
    # execute only if run as a script
    simulator = ORTSimulator()
    simulator.run()
