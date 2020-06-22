import numpy as np

from vispy import app, gloo
from vispy.util import keys

from circusort.io.probe import load_probe
from circusort.io.template import load_template_from_dict

from utils.widgets import Controler
from views.canvas import ViewCanvas, LinesPlot
from circusort.obj.cells import Cells
from circusort.obj.cell import Cell
from circusort.obj.train import Train
from circusort.obj.amplitude import Amplitude

    
RATES_VERT_SHADER = """
attribute float a_rate_value;
attribute float a_selected_cell;
attribute vec3 a_color;
attribute float a_index_cell;
attribute float a_index_t;


uniform vec2 u_scale;
uniform float u_max_value;
uniform float u_nb_points;

varying vec3 v_color;
varying float v_selected_cell;
varying float v_index_cell;
varying float v_pos_x;

void main() {
    float x = -0.9 + (1.8 * (a_index_t / u_nb_points) * u_scale.x);
    float y = -0.9 + a_rate_value/u_max_value;
    vec2 position = vec2(x, y);   
    gl_Position = vec4(position, 0.0, 1.0);
    v_color = a_color;
    v_index_cell = a_index_cell;
    v_pos_x = position.x;
    v_selected_cell = a_selected_cell;
}
"""

RATES_FRAG_SHADER = """
varying float v_pos_x;
varying float v_index_cell;
varying vec3 v_color;
varying float v_selected_cell;

// Fragment shader.
void main() {
    gl_FragColor = vec4(v_color, 1.0);
    if (v_selected_cell == 0.0)
        discard;
    if (fract(v_index_cell) > 0.0 || (v_pos_x) > 0.9)
        discard;
}
"""


class AmplitudeCanvas(ViewCanvas):

    requires = ['spikes', 'time']

    name = "Amplitudes"

    def __init__(self, probe_path=None, params=None):
        ViewCanvas.__init__(self, title="Amplitude view")

        self.probe = load_probe(probe_path)
        self.add_single_box()
        self.cells = Cells({})

        # Rates Shaders
        self.x_value = 0
        self.rate_mat = np.zeros((self.nb_templates, 30), dtype=np.float32)
        self.rate_vector = np.zeros(100).astype(np.float32)
        self.index_time, self.index_cell = 0, 0

        self.time_window = 50
        self.time_window_from_start = True
        self.list_selected_cells = []
        self.selected_cells_vector = 0

        self.u_scale = np.array([[1.0, 1.0]]).astype(np.float32)
        self.initialized = False

        self.programs['rates'] = ScatterPlot(RATES_VERT_SHADER, RATES_FRAG_SHADER)
        self.programs['rates']['a_rate_value'] = self.rate_vector

        # Final details.

        self.controler = AmplitudeControler(self)


    @property
    def nb_templates(self):
        return len(self.cells)

    def zoom_rates(self, zoom_value):
        self.u_scale = np.array([[zoom_value, 1.0]]).astype(np.float32)
        self.programs['rates']['u_scale'] = self.u_scale
        self.update()
        return

    def _highlight_selection(self, selection):
        self.list_selected_cells = [0] * self.nb_templates
        for i in selection:
            self.list_selected_cells[i] = 1
        self.selected_cells_vector = np.repeat(self.list_selected_cells, repeats=self.rate_mat.shape[1]).astype(
            np.float32)
        self.programs['rates']['a_selected_cell'] = self.selected_cells_vector
        return

    def _set_value(self, key, value):

        if key == "full":
            self.time_window_from_start = value
        elif key == "range":
            self.time_window = int((value[0] // value[1]))

    def _on_reception(self, data):
        
        spikes = data['spikes'] if 'spikes' in data else None
        self.time = data['time'] if 'time' in data else None
        old_size = self.nb_templates

        if spikes is not None:

            is_known = np.in1d(np.unique(spikes['templates']), self.cells.ids)
            not_kwown = is_known[is_known == False]

            for i in range(len(not_kwown)):
                template = None
                new_cell = Cell(template, Train([], t_min=0), Amplitude([], [], t_min=0))
                self.cells.append(new_cell)

            if self.initialized is False:
                self.list_selected_cells = [1] * self.nb_templates
                self.initialized = True
            else:
                for i in range(self.nb_templates - old_size):
                    self.list_selected_cells.append(0)

            self.cells.add_spikes(spikes['spike_times'], spikes['amplitudes'], spikes['templates'])    
            self.cells.set_t_max(self.time)
            rates = self.cells.rate(self.controler.bin_size)

            if self.time_window_from_start is True:
                self.rate_mat = rates
            else:
                self.rate_mat = rates[:, -self.time_window:]

            self.rate_vector = self.rate_mat.ravel().astype(np.float32)

            self.selected_cells_vector = np.repeat(self.list_selected_cells, repeats=self.rate_mat.shape[1]).astype(
                np.float32)
            self.index_time = np.tile(np.arange(0, self.rate_mat.shape[1], dtype=np.float32), reps=self.nb_templates)
            self.index_cell = np.repeat(np.arange(0, self.nb_templates, dtype=np.float32),
                                        repeats=self.rate_mat.shape[1])
            
            self.color_rates = np.repeat(self.get_colors(self.nb_templates), repeats=self.rate_mat.shape[1], axis=0)

            self.programs['rates']['a_rate_value'] = self.rate_vector
            self.programs['rates']['u_max_value'] = np.amax(self.rate_vector)
            self.programs['rates']['a_selected_cell'] = self.selected_cells_vector
            self.programs['rates']['a_color'] = self.color_rates
            self.programs['rates']['a_index_t'] = self.index_time
            self.programs['rates']['a_index_cell'] = self.index_cell
            self.programs['rates']['u_scale'] = self.u_scale
            self.programs['rates']['u_nb_points'] = self.rate_mat.shape[1]
        return


class RateControler(Controler):

    def __init__(self, canvas, bin_size=0.1):
        '''
        Control widgets:
        '''

        Controler.__init__(self, canvas)
        self.bin_size = bin_size
        
        self.dsb_bin_size = self.double_spin_box(label='Bin Size', unit='seconds', min_value=0.01,
                                                 max_value=100, step=0.1, init_value=self.bin_size)

        self.dsb_zoom = self.double_spin_box(label='Zoom', min_value=1, max_value=50, step=0.1,
                                             init_value=1)
        self.dsb_time_window = self.double_spin_box(label='Time window', unit='seconds',
                                                    min_value=1, max_value=50, step=0.1,
                                                    init_value=1)
        self.cb_tw = self.checkbox(label='Time window from start', init_state=True)


        self.add_widget(self.dsb_bin_size, self._on_binsize_changed)
        self.add_widget(self.dsb_zoom, self._on_zoomrates_changed)
        self.add_widget(self.dsb_time_window, self._on_time_window_changed)
        self.add_widget(self.cb_tw, self._time_window_rate_full)

    def _on_binsize_changed(self, bin_size):
        self.bin_size = self.dsb_bin_size['widget'].value()
        return

    def _on_zoomrates_changed(self):
        zoom_value = self.dsb_zoom['widget'].value()
        self.canvas.zoom_rates(zoom_value)
        return

    def _time_window_rate_full(self):
        value = self.cb_tw['widget'].isChecked()
        self.canvas.set_value({"full" : value})
        return

    def _on_time_window_changed(self):
        tw_value = self.dsb_time_window['widget'].value()
        self.canvas.set_value({"range" : (tw_value, self.bin_size)})
        return
