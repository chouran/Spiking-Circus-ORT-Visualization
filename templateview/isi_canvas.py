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


class ISICanvas(app.Canvas):

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

        self.initialized = False
        self.nb_points, self.nb_cells = 0, 0
        self.list_selected_isi, self.selected_isi_vector = [], 0
        self.list_isi, self.isi_vector = 0, 5
        self.index_x, self.index_cell, self.color_isi = 1, 0, 0
        self.u_scale = [1.0, 1.0]

        self.isi_program = gloo.Program(vert=ISI_VERT_SHADER, frag=ISI_FRAG_SHADER)
        self.isi_program['a_isi_value'] = self.isi_vector
        self.isi_program['a_index_x'] = self.index_x
        self.isi_program['u_scale'] = self.u_scale

        # Final details.
        gloo.set_viewport(0, 0, *self.physical_size)
        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    @staticmethod
    def on_resize_quatro(event):
        gloo.set_viewport(0, 0, *event.physical_size)
        return

    def on_draw(self, event):
        __ = event
        gloo.clear()
        self._box_program.draw('line_strip')
        self.isi_program.draw('line_strip')
        return

    def zoom_isi(self, zoom_value):
        self.u_scale = np.array([[zoom_value, 1.0]]).astype(np.float32)
        self.isi_program['u_scale'] = self.u_scale
        self.update()
        return

    # TODO : Selection templates
    def selected_cells(self, l_select):
        self.list_selected_isi = [0] * self.nb_cells
        for i in l_select:
            self.list_selected_isi[i] = 1
        self.selected_isi_vector = np.repeat(self.list_selected_isi, repeats=self.nb_points).astype(np.float32)
        self.isi_program['a_selected_cell'] = self.selected_isi_vector
        self.update()
        return

    def on_reception_isi(self, isi):
        if isi is not None and len(list(isi)) != 0:
            if self.initialized is False:
                self.nb_points = isi[0][1].shape[0]
                self.nb_cells = len(list(isi))
                self.list_selected_isi = [1] * self.nb_cells
                self.initialized = True

            else:
                if self.nb_cells != len(list(isi)):
                    for i in range(len(list(isi)) - self.nb_cells):
                        self.list_selected_isi.append(1)
                    self.nb_cells = len(list(isi))

            list_isi_values = []
            for i in list(isi):
                list_isi_values.append(isi[i][1])

            self.list_isi = [y for x in list_isi_values for y in x]
            self.isi_vector = np.array(self.list_isi, dtype=np.float32)
            self.selected_isi_vector = np.repeat(self.list_selected_isi, repeats=self.nb_points).astype(
                np.float32)
            self.index_x = np.tile(np.arange(0, self.nb_points), reps=self.nb_cells).astype(np.float32)
            self.index_cell = np.repeat(np.arange(0, self.nb_cells), repeats=self.nb_points).astype(np.float32)
            np.random.seed(12)
            colors = np.random.uniform(size=(self.nb_cells, 3), low=0.3, high=.99).astype(np.float32)
            self.color_isi = np.repeat(colors, repeats=self.nb_points, axis=0)

            self.isi_program['a_isi_value'] = self.isi_vector
            self.isi_program['a_selected_cell'] = self.selected_isi_vector
            self.isi_program['a_color'] = self.color_isi
            self.isi_program['a_index_x'] = self.index_x
            self.isi_program['a_index_cell'] = self.index_cell
            self.isi_program['u_scale'] = self.u_scale
            self.isi_program['u_nb_points'] = self.nb_points
            self.isi_program['u_max_value'] = np.amax(self.isi_vector)

            self.update()
        return
