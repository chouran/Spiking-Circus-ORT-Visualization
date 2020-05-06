from circusort.io.probe import load_probe
from circusort.io.template_store import load_template_store
from circusort.io.cells import load_cells
from circusort.io.spikes import load_spikes, spikes2cells
from circusort.obj.cells import Cells
from circusort.obj.cell import Cell

templates = load_template_store('data/templates.h5', 'probe.prb')
fitted_spikes = load_spikes('data/spikes.h5')