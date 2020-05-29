import numpy as np

from vispy import app, gloo
from vispy.util import keys

from circusort.io.probe import load_probe
from circusort.io.template import load_template_from_dict

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
attribute vec3 a_color;
attribute float a_index_cell;
attribute float a_index_time;
attribute float a_index_bar;

uniform float u_nb_cells;
uniform float u_nb_cells_selected;
uniform float u_scale_x;
uniform float u_max_value;

varying vec3 v_color;
varying float v_index_cell;
varying float v_pos_x;

void main() {
    float w_t = 1.8/u_scale_x;
    float w_bar = w_t/(u_nb_cells_selected+2);

    float y = -0.9 + a_rate_value/u_max_value;
    float x = -(0.9) + w_t*a_index_time + w_bar * (a_index_cell+a_index_bar+2);

    gl_Position = vec4(x, y, 0.0, 1.0);

    v_color = a_color;
    v_index_cell = a_index_cell;
    v_pos_x = x;
}
"""

RATES_FRAG_SHADER = """
varying float v_pos_x;
varying float v_index_cell;
varying vec3 v_color;

// Fragment shader.
void main() {
    gl_FragColor = vec4(v_color, 1.0);
    if (fract(v_index_cell) > 0.0 || (v_pos_x) > 0.9)
        discard;
}
"""


class RateCanvas(app.Canvas):

    def __init__(self, probe_path=None, params=None):
        app.Canvas.__init__(self, title="Vispy canvas2")

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
        self.nb_cells, self.nb_cells_selected, self.time_max = 0, 0, 0
        self.rate_vector = np.zeros(100).astype(np.float32)
        self.color_rates, self.index_bar, self.index_time = 0, 0, 0
        self.rate_mat = np.zeros((self.nb_cells, 30), dtype=np.float32)
        self.scale_x = 20
        self.nb_cells, self.time_max = 0, 0
        self.index_cell, self.index_cell_selec = 0, 0
        self.list_unselected_cells = list(range(self.nb_cells))
        self.list_selected_cells = []
        self.initialized = True

        self.rates_program = gloo.Program(vert=RATES_VERT_SHADER, frag=RATES_FRAG_SHADER)
        self.rates_program['a_rate_value'] = self.rate_vector
        # self.rates_program['a_rate_value'] = y_data
        # self.rates_program['a_color'] = color_bar
        # self.rates_program['a_index_time'] = index_time
        # self.rates_program['a_index_cell'] = index_cell
        # self.rates_program['a_index_x'] = index_x

        # self.rates_program['u_time_max'] = self.time_max
        # self.rates_program['u_nb_points'] = self.nb_points

        gloo.set_viewport(0, 0, *self.physical_size)
        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

        # Final details.

        gloo.set_viewport(0, 0, *self.physical_size)
        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    def on_resize_bis(event):
        gloo.set_viewport(0, 0, *event.physical_size)
        return

    def on_draw(self, event):
        __ = event
        gloo.clear()
        gloo.set_viewport(0, 0, *self.physical_size)
        self._box_program.draw('line_strip')
        self.rates_program.draw('triangle_strip')
        return

    def zoom_axis_t(self, zoom_value):
        self.scale_x = zoom_value
        self.rates_program['u_scale_x'] = self.scale_x
        self.update()
        return

    # TODO : Selection templates
    def selected_cells(self, l_select):
        self.nb_cells_selected = len(l_select)
        self.list_selected_cells = l_select

        for i in range(self.nb_cells):
            if i in self.list_selected_cells is True:
                self.list_unselected_cells.remove(i)

        index_cell = np.zeros(self.nb_cells, dtype=np.float32)
        index_nb = 1
        for j in self.list_selected_cells:
            index_cell[j] = index_nb
            index_nb += 1

        self.index_cell_selec = np.tile(np.repeat(index_cell, repeats=4), reps=self.rate_mat.shape[1])
        self.rates_program['a_index_cell'] = self.index_cell_selec
        self.update()
        return

    def on_reception_rates(self, rates):
        if rates is not None:
            if rates.shape[0] != 0:
                self.nb_cells = rates.shape[0]
                k = 50

                if rates.shape[0] != self.nb_cells:
                    self.nb_cells = rates.shape[0]
                    index_new_cell = np.zeros(self.nb_cells, dtype=np.float32)
                    index_nb = 1
                    for j in self.list_selected_cells:
                        index_cell[j] = index_nb
                        index_nb += 1

                    self.index_cell = np.tile(np.repeat(index_cell, repeats=4), reps=self.rate_mat.shape[1])

                if not self.initialized:
                    self.nb_cells = rates.shape[0]
                    self.rate_mat = np.zeros((self.nb_cells, 30), dtype=np.float32)

                    self.initialized = True

                else:
                    self.list_cells = list(range(self.nb_cells))

                    # Refresh the rate matrix with the new values
                    # self.rate_mat[:, :-k] = self.rate_mat[:, k:]
                    # self.rate_mat[:, -k:] = np.reshape(rates[:, -1], (self.nb_cells, 1))
                    self.rate_mat = rates[:, -k:].astype(np.float32)

                    # Construct the vector for the bar plot
                    y_zeros = np.zeros(2 * self.nb_cells * self.rate_mat.shape[1]).astype(np.float32)
                    self.rate_vector = np.c_[y_zeros, np.repeat(self.rate_mat.ravel(order='F'),
                                                                repeats=2)].ravel()

                self.index_bar = np.tile(np.array([0, 0, 1, 1], dtype=np.float32),
                                         reps=self.nb_cells * self.rate_mat.shape[1])
                self.index_time = np.repeat(np.arange(0, self.rate_mat.shape[1]).astype(np.float32),
                                            repeats=4 * self.nb_cells)
                self.index_cell = np.tile(np.repeat(np.arange(1, self.nb_cells + 1).astype(np.float32), repeats=4),
                                          reps=self.rate_mat.shape[1])

                self.rates_program['a_rate_value'] = self.rate_vector
                self.rates_program['a_color'] = self.color_rates
                self.rates_program['a_index_bar'] = self.index_bar
                self.rateslis_program['a_index_time'] = self.index_time
                self.rates_program['a_index_cell'] = self.index_cell
                self.rates_program['u_nb_cells'] = self.nb_cells
                self.rates_program['u_nb_cells_selected'] = self.nb_cells
                self.rates_program['u_scale_x'] = self.scale_x
                self.rates_program['u_max_value'] = 20

            self.update()

        return

