import numpy as np

from vispy import app, gloo
from vispy.util import keys

from circusort.io.probe import load_probe
from circusort.io.template import load_template_from_dict

from utils.widgets import Controler
from views.canvas import ViewCanvas
from views.programs import LinesPlot, SingleLinePlot, SingleScatterPlot
from circusort.obj.cells import Cells
from circusort.obj.cell import Cell
from circusort.obj.train import Train
from circusort.obj.amplitude import Amplitude

class AmplitudeCanvas(ViewCanvas):

    requires = ['spikes', 'time']
    name = "Amplitudes"

    def __init__(self, probe_path=None, params=None):

        ViewCanvas.__init__(self, probe_path, title="Amplitude view", box='single')
        self.cells = Cells({})
        self.programs['amplitudes'] = SingleScatterPlot(nb_dimensions=2)

    @property
    def nb_templates(self):
        return len(self.cells)

    def _highlight_selection(self, selection):
        self.programs['amplitudes'].set_selection(selection)
        return

    def _on_reception(self, data):

        self.time = data['time'] if 'time' in data else None
        spikes = data['spikes'] if 'spikes' in data else None

        if spikes is not None:

            is_known = np.in1d(np.unique(spikes['templates']), self.cells.ids)
            not_kwown = is_known[is_known == False]

            for i in range(len(not_kwown)):
                new_cell = Cell(None, Train([], t_min=0), Amplitude([], [], t_min=0))
                self.cells.append(new_cell)

            self.cells.add_spikes(spikes['spike_times'], spikes['amplitudes'], spikes['templates'])
            colors = self.get_colors(self.nb_templates)
            self.programs['amplitudes'].set_data([self.cells.spikes, self.cells.amplitudes], self.time, colors)

        self.update()

        return