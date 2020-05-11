import numpy as np
from multiprocessing import Pipe
from gui_process import GUIProcess
from circusort.io.probe import load_probe
from circusort.io.template_store import load_template_store
from circusort.io.spikes import load_spikes

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
        self.spikes = load_spikes('data/spikes.h5')

        self._params_pipe = Pipe()
        self._number_pipe = Pipe()
        self._templates_pipe = Pipe()
        self._spikes_pipe = Pipe()
        self._qt_process = GUIProcess(self._params_pipe, self._number_pipe, self._templates_pipe, self._spikes_pipe,
                                      probe_path=self.probe_path)

        self._qt_process.start()
        self.number = self.templates[0].creation_time - 10
        self.index = 0

        self._params_pipe[1].send({
            'nb_samples': self.nb_samples,
            'sampling_rate': self.sampling_rate,
        })

        return

    def run(self):

        while True:
            # Here we are increasing the counter
            self.number += 1
            if self.number == self.templates[self.index].creation_time:
                templates = [self.templates[self.index].to_dict()]
                self.index += 1
            else:
                templates = None
            
            t_min = (self.number - 1)*self.nb_samples / self.sampling_rate
            t_max = self.number*self.nb_samples / self.sampling_rate

            # If we want to send real spikes
            spikes = self.spikes.get_spike_data(t_min, t_max, range(self.index))

            self._number_pipe[1].send(self.number)
            self._templates_pipe[1].send(templates)
            self._spikes_pipe[1].send(spikes)

if __name__ == "__main__":
    # execute only if run as a script
    simulator = ORTSimulator()
    simulator.run()
