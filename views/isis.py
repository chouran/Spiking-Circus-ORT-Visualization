import numpy as np
import scipy.signal

from vispy import app, gloo
from vispy.util import keys

from circusort.io.probe import load_probe
from circusort.io.template import load_template_from_dict

from utils.widgets import Controler

from views.canvas import ViewCanvas


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
    float y = -0.9 + 1.8*a_isi_value/u_max_value ;
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

    requires = ['isis']
    name = "ISIs"

    def __init__(self, probe_path=None, params=None):
        ViewCanvas.__init__(self, title="ISI view")

        self.probe = load_probe(probe_path)
        self.add_single_box()
        # self.channels = params['channels']
        self.nb_channels = self.probe.nb_channels
        self.init_time = 0

        self.initialized = False
        self.nb_points, self.nb_cells = 0, 0
        self.list_selected_isi, self.selected_isi_vector = [], 0
        self.list_isi, self.isi_vector = 0, 5
        self.index_x, self.index_cell, self.color_isi = 1, 0, 0
        self.isi_mat, self.isi_smooth = 0, 0
        self.u_scale = [1.0, 1.0]

        self.programs['isis'] = gloo.Program(vert=ISI_VERT_SHADER, frag=ISI_FRAG_SHADER)
        self.programs['isis']['a_isi_value'] = self.isi_vector
        self.programs['isis']['a_index_x'] = self.index_x
        self.programs['isis']['u_scale'] = self.u_scale

        self.controler = ISIControler(self)

    def zoom_isi(self, zoom_value):
        self.u_scale = np.array([[zoom_value, 1.0]]).astype(np.float32)
        self.programs['isis']['u_scale'] = self.u_scale
        self.update()
        return

    # TODO : Selection templates
    def _highlight_selection(self, selection):
        self.list_selected_isi = [0] * self.nb_cells
        for i in selection:
            self.list_selected_isi[i] = 1
        self.selected_isi_vector = np.repeat(self.list_selected_isi, repeats=self.nb_points).astype(np.float32)
        self.programs['isis']['a_selected_cell'] = self.selected_isi_vector
        return

    def _on_reception(self, data):
        isi = data['rates']
        if isi is not None and len(list(isi)) != 0:
            if self.initialized is False:
                self.nb_points = isi[0][0].shape[0]
                self.nb_cells = len(list(isi))
                self.list_selected_isi = [1] * self.nb_cells
                self.initialized = True

            else:
                if self.nb_cells != len(list(isi)):
                    for i in range(len(list(isi)) - self.nb_cells):
                        self.list_selected_isi.append(0)
                    self.nb_cells = len(list(isi))

            list_isi_values = []
            for i in list(isi):
                list_isi_values.append(isi[i][0])

            self.list_isi = [y for x in list_isi_values for y in x]
            self.isi_vector = np.array(self.list_isi, dtype=np.float32)
            self.isi_mat = np.reshape(self.isi_vector, (self.nb_cells, self.nb_points))
            #print(self.isi_mat.shape)
            self.isi_smooth = (scipy.signal.savgol_filter(self.isi_mat, 5, 3, axis=1)).ravel()

            self.selected_isi_vector = np.repeat(self.list_selected_isi, repeats=self.nb_points).astype(
                np.float32)
            self.index_x = np.tile(np.arange(0, self.nb_points), reps=self.nb_cells).astype(np.float32)
            self.index_cell = np.repeat(np.arange(0, self.nb_cells), repeats=self.nb_points).astype(np.float32)
            self.color_isi = np.repeat(self.get_colors(self.nb_templates), repeats=self.nb_points, axis=0)

            self.programs['isis']['a_isi_value'] = self.isi_smooth
            self.programs['isis']['a_selected_cell'] = self.selected_isi_vector
            self.programs['isis']['a_color'] = self.color_isi
            self.programs['isis']['a_index_x'] = self.index_x
            self.programs['isis']['a_index_cell'] = self.index_cell
            self.programs['isis']['u_scale'] = self.u_scale
            self.programs['isis']['u_nb_points'] = self.nb_points
            self.programs['isis']['u_max_value'] = np.amax(self.isi_vector)

        return

class ISIControler(Controler):

    def __init__(self, canvas, bin_size=0.02, max_time=1):
        '''
        Control widgets:
        '''

        # TODO ISI

        Controler.__init__(self, canvas)
        self.bin_size = bin_size 
        self.max_time = max_time

        self.dsb_bin_size = self.double_spin_box(label='Bin Size', unit='seconds', min_value=0.001,
                                                 max_value=0.1, step=0.001, init_value=self.bin_size)

        self.dsb_zoom = self.double_spin_box(label='Zoom', min_value=1, max_value=50, step=0.1,
                                             init_value=1)

        self.dsb_time_window = self.double_spin_box(label='Max time', unit='seconds',
                                                    min_value=0, max_value=50, step=0.1,
                                                    init_value=0.25)

        self.add_widget(self.dsb_bin_size, self._on_binsize_changed)
        self.add_widget(self.dsb_zoom, self._on_zoomrates_changed)
        self.add_widget(self.dsb_time_window, self._on_time_changed)


    def _on_binsize_changed(self, bin_size):
        time_bs = self.dsb_bin_size['widget'].value()
        bin_size_obj = time_bs
        self.dsb_time_window['widget'].setSingleStep(time_bs)
        return

    def _on_zoomrates_changed(self):
        zoom_value = self.dsb_zoom['widget'].value()
        self.canvas.zoom_rates(zoom_value)
        return

    def _on_time_changed(self):
        tw_value = self._dsp_tw_rate.value()
        self.canvas.set_value("max_time", self.max_time)
        return