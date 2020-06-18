import numpy as np

from vispy import app, gloo
from vispy.util import keys

from circusort.io.probe import load_probe
from circusort.io.template import load_template_from_dict

import utils.widgets as wid

from views.canvas import ViewCanvas

BOX_VERT_SHADER = """
attribute vec2 a_position;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""
BOX_FRAG_SHADER = """
// Fragment shader.
void main() {
    gl_FragColor = vec4(0.5, 0.5, 0.5, 1.0);
}
"""

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


class RateCanvas(ViewCanvas):

    requires = ['rates']

    def __init__(self, probe_path=None, params=None):
        ViewCanvas.__init__(self, title="Rate view")

        self.probe = load_probe(probe_path)
        # self.channels = params['channels']
        self.nb_channels = self.probe.nb_channels
        self.init_time = 0

        box_corner_positions = np.array([[+0.9, +0.9],
                                         [-0.9, +0.9],
                                         [-0.9, -0.9],
                                         [+0.9, -0.9],
                                         [+0.9, +0.9]], dtype=np.float32)

        self._box_program = gloo.Program(vert=BOX_VERT_SHADER, frag=BOX_FRAG_SHADER)
        self._box_program['a_position'] = box_corner_positions

        # Rates Shaders
        self.x_value = 0
        self.nb_cells = 0
        self.rate_mat = np.zeros((self.nb_cells, 30), dtype=np.float32)
        self.rate_vector = np.zeros(100).astype(np.float32)
        self.index_time, self.index_cell = 0, 0
        self.color_rates = np.array([[1, 1, 1]])

        self.time_window = 50
        self.time_window_from_start = True
        self.list_selected_cells = []
        self.selected_cells_vector = 0
        self.rate_mat_cum, self.rate_vector_cum = 0, 0
        self.u_scale = np.array([[1.0, 1.0]]).astype(np.float32)
        self.initialized = False
        self.cum_plots = False

        self.rates_program = gloo.Program(vert=RATES_VERT_SHADER, frag=RATES_FRAG_SHADER)
        self.rates_program['a_rate_value'] = self.rate_vector

        # Final details.

        gloo.set_viewport(0, 0, *self.physical_size)
        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    def on_draw(self, event):
        __ = event
        gloo.clear()
        gloo.set_viewport(0, 0, *self.physical_size)
        self._box_program.draw('line_strip')
        self.rates_program.draw('line_strip')
        return

    def zoom_rates(self, zoom_value):
        self.u_scale = np.array([[zoom_value, 1.0]]).astype(np.float32)
        self.rates_program['u_scale'] = self.u_scale
        self.update()
        return

    def _highlight_selection(self, selection):
        self.list_selected_cells = [0] * self.nb_cells
        for i in selection:
            self.list_selected_cells[i] = 1
        self.selected_cells_vector = np.repeat(self.list_selected_cells, repeats=self.rate_mat.shape[1]).astype(
            np.float32)
        self.rates_program['a_selected_cell'] = self.selected_cells_vector
        return

    def _set_value(self, key, value):

        if key == "full":
            self.time_window_from_start = value
        elif key == "range":
            self.time_window = int((value[0] // value[1]))

    def _on_reception(self, data):
        rates = data['rates']
        if rates is not None and rates.shape[0] != 0:
            if self.initialized is False:
                self.nb_cells = rates.shape[0]
                self.list_selected_cells = [1] * self.nb_cells
                self.rate_mat = rates
                self.initialized = True
            else:
                if self.nb_cells != rates.shape[0]:
                    for i in range(rates.shape[0] - self.nb_cells):
                        self.list_selected_cells.append(0)
                    self.nb_cells = rates.shape[0]

            if self.time_window_from_start is True:
                self.rate_mat = rates
            else:
                self.rate_mat = rates[:, -self.time_window:]
                # k = 50
                # self.rate_mat = rates[:, -k:].astype(np.float32)

            self.rate_vector = self.rate_mat.ravel().astype(np.float32)

            self.selected_cells_vector = np.repeat(self.list_selected_cells, repeats=self.rate_mat.shape[1]).astype(
                np.float32)
            self.index_time = np.tile(np.arange(0, self.rate_mat.shape[1], dtype=np.float32), reps=self.nb_cells)
            self.index_cell = np.repeat(np.arange(0, self.nb_cells, dtype=np.float32),
                                        repeats=self.rate_mat.shape[1])
            np.random.seed(12)
            colors = np.random.uniform(size=(self.nb_cells, 3), low=0.3, high=.99).astype(np.float32)
            self.color_rates = np.repeat(colors, repeats=self.rate_mat.shape[1], axis=0)

            self.rates_program['a_rate_value'] = self.rate_vector
            self.rates_program['u_max_value'] = np.amax(self.rate_vector)
            self.rates_program['a_selected_cell'] = self.selected_cells_vector
            self.rates_program['a_color'] = self.color_rates
            self.rates_program['a_index_t'] = self.index_time
            self.rates_program['a_index_cell'] = self.index_cell
            self.rates_program['u_scale'] = self.u_scale
            self.rates_program['u_nb_points'] = self.rate_mat.shape[1]
        return


class RateControl(wid.CustomWidget):
    def __init__(self, rate_canv_obj, bin_size_obj):
        '''
        Control widgets:
        '''

        self.bin_size = 1.0

        self.dsb_bin_size = self.double_spin_box(label='Bin Size', unit='seconds', min_value=0.1,
                                                 max_value=10, step=0.1, init_value=bin_size_obj)
        self.dsb_zoom = self.double_spin_box(label='Zoom', min_value=1, max_value=50, step=0.1,
                                             init_value=1)
        self.dsb_time_window = self.double_spin_box(label='Time window', unit='seconds',
                                                    min_value=1, max_value=50, step=0.1,
                                                    init_value=1)
        self.cb_tw = self.checkbox(label='Time window from start', init_state=True)

        ### Create the dock widget to be added in the QT window docking space
        self.dock_widget = wid.dock_control('Rate View Params', 'Left', self.dsb_bin_size,
                                            self.dsb_time_window, self.cb_tw,
                                            self.dsb_zoom)

        ### Signals
        self.dsb_bin_size['widget'].valueChanged.connect(lambda: self._on_binsize_changed(bin_size_obj))
        self.dsb_zoom['widget'].valueChanged.connect(lambda: self._on_zoomrates_changed(rate_canv_obj))
        self.dsb_time_window['widget'].valueChanged.connect(lambda: self._on_time_changed(rate_canv_obj))
        self.cb_tw['widget'].stateChanged.connect(lambda: self._time_window_rate_full(rate_canv_obj))

    # -----------------------------------------------------------------------------
    # Signals methods
    # -----------------------------------------------------------------------------

    def _on_binsize_changed(self, bin_size_obj):
        time_bs = self.dsb_bin_size['widget'].value()
        bin_size_obj = time_bs
        self.dsb_time_window['widget'].setSingleStep(time_bs)

        return

    def _on_zoomrates_changed(self, rate_canv):
        zoom_value = self.dsb_zoom['widget'].value()
        rate_canv.zoom_rates(zoom_value)

        return

    def _time_window_rate_full(self, rate_canv):
        value = self.cb_tw['widget'].isChecked()
        rate_canv.time_window_full(value)

        return

    def _on_time_window_changed(self, rate_canv):
        tw_value = self._dsp_tw_rate.value()
        rate_canv.time_window_value(tw_value, self.bin_size)
        return
