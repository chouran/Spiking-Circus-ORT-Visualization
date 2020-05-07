from circusort.io.probe import load_probe
from circusort.io.template_store import load_template_store
from circusort.io.cells import load_cells
from circusort.io.spikes import load_spikes, spikes2cells
from circusort.obj.cells import Cells
from circusort.obj.cell import Cell
from circusort.obj.train import Train
from circusort.obj.amplitude import Amplitude

templates = load_template_store('data/templates.h5', 'probe.prb')
fitted_spikes = load_spikes('data/spikes.h5')



# spikes_a = fitted_spikes.get_spike_data(t_min=0, t_max=100, indices=[0])
# train = Train(spikes_a['spike_times'])
# amplitude = Amplitude(spikes_a['amplitudes'], spikes_a['spike_times'])

b = Cells({})

for i in range(len(templates)):
    mytemplate = templates[i]
    new_cell = Cell(mytemplate, Train([]), Amplitude([], []))
    b.append(new_cell)


# b.append(a)


# spikes_b = fitted_spikes.get_spike_data(t_min=100, t_max=120, indices=[0])
# b.add_spikes(spikes_b['spike_times'], spikes_b['amplitudes'], spikes_b['templates'])

spikes_all = fitted_spikes.get_spike_data()
b.add_spikes(spikes_all['spike_times'], spikes_all['amplitudes'], spikes_all['templates'])
b.set_t_max(150)

# train = Train(spikes_b['spike_times'])
# amplitude = Amplitude(spikes_b['amplitudes'], spikes_b['spike_times'])
# a.add_spikes(train, amplitude)