import numpy as np
import scipy.signal

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

ISI_VERT_SHADER = """
attribute float a_isi_value;
attribute float a_selected_cell;
attribute vec3 a_color;
attribute float a_index_cell;
attribute float a_index_x;

uniform vec2 u_scale;
uniform float u_max_value;
uniform float u_nb_points;

varying vec3 v_color;
varying float v_selected_cell;
varying float v_index_cell;
varying float v_pos_x;

void main() {
    float x = -0.9 + (1.8 * (a_index_x / u_nb_points)) ;
    float y = -0.9 + a_isi_value/u_max_value ;
    vec2 position = vec2(x - (1 - 1/u_scale.x), y);   
    gl_Position = vec4(position, 0.0, 1.0);
    v_color = a_color;
    v_index_cell = a_index_cell;
    v_pos_x = position.x;
    v_selected_cell = a_selected_cell;
}
"""

ISI_FRAG_SHADER = """
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


class ISICanvas(ViewCanvas):

    requires = ['spikes', 'time']
    name = "ISIs"

    def __init__(self, probe_path=None, params=None):
        ViewCanvas.__init__(self, title="ISI view")

        self.probe = load_probe(probe_path)
        self.add_single_box()
        self.cells = Cells({})

        self.initialized = False
        self.list_selected_cells, self.selected_isi_vector = [], 0
        self.index_x, self.index_cell = 1, 0
        self.isi_mat, self.isi_smooth = 0, 0
        self.isi_vector = np.zeros(100).astype(np.float32)
        self.u_scale = [1.0, 1.0]

        self.programs['isis'] = LinesPlot(vert=ISI_VERT_SHADER, frag=ISI_FRAG_SHADER)
        self.programs['isis']['a_isi_value'] = self.isi_vector

        self.controler = ISIControler(self)

    @property
    def nb_templates(self):
        return len(self.cells)

    def zoom_isi(self, zoom_value):
        self.u_scale = np.array([[zoom_value, 1.0]]).astype(np.float32)
        self.programs['isis']['u_scale'] = self.u_scale
        self.update()
        return

    # TODO : Selection templates
    def _highlight_selection(self, selection):
        self.list_selected_cells = [0] * self.nb_templates
        for i in selection:
            self.list_selected_cells[i] = 1
        self.selected_isi_vector = np.repeat(self.list_selected_cells, repeats=self.nb_points).astype(np.float32)
        self.programs['isis']['a_selected_cell'] = self.selected_isi_vector
        return

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
            
            all_isis = self.cells.interspike_interval_histogram(self.controler.bin_size, self.controler.max_time)
            self.isi_vector = np.array([isi[0] for isi in all_isis.values()]).astype(np.float32)
            
            if len(self.isi_vector) > 0:
                self.nb_points = self.isi_vector.shape[1]
                self.max_isi = np.amax(self.isi_vector)
            else:
                self.nb_points = 0
                self.max_isi = 0

            self.selected_isi_vector = np.repeat(self.list_selected_cells, repeats=self.nb_points).astype(
                np.float32)

            self.index_x = np.tile(np.arange(self.nb_points), reps=self.nb_templates).astype(np.float32)
            self.index_cell = np.repeat(np.arange(self.nb_templates), repeats=self.nb_points).astype(np.float32)
            self.color_isi = np.repeat(self.get_colors(self.nb_templates), repeats=self.nb_points, axis=0)

            self.programs['isis']['a_isi_value'] = self.isi_vector.ravel()
            self.programs['isis']['a_selected_cell'] = self.selected_isi_vector
            self.programs['isis']['a_color'] = self.color_isi
            self.programs['isis']['a_index_x'] = self.index_x
            self.programs['isis']['a_index_cell'] = self.index_cell
            self.programs['isis']['u_scale'] = self.u_scale
            self.programs['isis']['u_nb_points'] = self.nb_points
            self.programs['isis']['u_max_value'] = self.max_isi

        return

class ISIControler(Controler):

    def __init__(self, canvas, bin_size=0.1, max_time=2.5):
        '''
        Control widgets:
        '''

        # TODO ISI

        Controler.__init__(self, canvas)
        self.bin_size = bin_size 
        self.max_time = max_time

        self.dsb_bin_size = self.double_spin_box(label='Bin Size', unit='seconds', min_value=0.001,
                                                 max_value=1, step=0.001, init_value=self.bin_size)

        self.dsb_zoom = self.double_spin_box(label='Zoom', min_value=1, max_value=50, step=0.1,
                                             init_value=1)

        self.dsb_time_window = self.double_spin_box(label='Max time', unit='seconds',
                                                    min_value=0, max_value=50, step=0.1,
                                                    init_value=self.max_time)

        self.add_widget(self.dsb_bin_size, self._on_binsize_changed)
        self.add_widget(self.dsb_zoom, self._on_zoomrates_changed)
        self.add_widget(self.dsb_time_window, self._on_time_changed)

    def _on_binsize_changed(self, bin_size):
        self.bin_size = self.dsb_bin_size['widget'].value()
        return

    def _on_zoomrates_changed(self):
        zoom_value = self.dsb_zoom['widget'].value()
        self.canvas.zoom_rates(zoom_value)
        return

    def _on_time_changed(self):
        self.max_time = self.dsb_time_window['widget'].value()
        return