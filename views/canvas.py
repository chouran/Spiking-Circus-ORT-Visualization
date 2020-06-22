import utils.widgets as wid
import numpy as np

from vispy import app, gloo
from vispy.util import keys

SINGLE_BOX_VERT_SHADER = """
attribute vec2 a_position;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

SINGLE_BOX_FRAG_SHADER = """
// Fragment shader.
void main() {
    gl_FragColor = vec4(0.5, 0.5, 0.5, 1.0);
}
"""

MULTI_BOX_FRAG_SHADER = """
// Varying variable.
varying float v_index;
// Fragment shader.
void main() {
    gl_FragColor = vec4(0.25, 0.25, 0.25, 1.0);
    // Discard the fragments between the box (emulate glMultiDrawArrays).
    if (fract(v_index) > 0.0) 
        discard;
}
"""

MULTI_BOX_VERT_SHADER = """
// Index of the box.
attribute float a_box_index;
// Coordinates of the position of the box.
attribute vec2 a_box_position;
// Coordinates of the position of the corner.
attribute vec2 a_corner_position;
// Uniform variables used to transform the subplots.
uniform float u_x_min;
uniform float u_x_max;
uniform float u_y_min;
uniform float u_y_max;
uniform float u_d_scale;
// Varying variable used for clipping in the fragment shader.
varying float v_index;
// Vertex shader.
void main() {
    // Compute the x coordinate.
    float x = a_corner_position.x;
    // Compute the y coordinate.
    float y = a_corner_position.y;
    // Compute the position.
    vec2 p = a_corner_position;
    // Find the affine transformation for the subplots.
    float w = u_x_max - u_x_min;
    float h = u_y_max - u_y_min;
    vec2 a = vec2(1.0 / (1.0 + w / u_d_scale), 1.0 / (1.0 + h / u_d_scale));
    vec2 b = vec2(-1.0 + 2.0 * (a_box_position.x - u_x_min) / w, -1.0 + 2.0 * (a_box_position.y - u_y_min) / h);
    // Apply the transformation.
    gl_Position = vec4(a * p + b, 0.0, 1.0);
    v_index = a_box_index;
}
"""

class LinesPlot(gloo.Program):

    def __init__(self, vert, frag):
        gloo.Program.__init__(self, vert=vert, frag=frag)

    def _draw(self):

        self.draw('line_strip')

class ScatterPlot(gloo.Program):

    def __init__(self, vert, frag):
        gloo.Program.__init__(self, vert=vert, frag=frag)

    def _draw(self):

        self.draw('points')

class ViewCanvas(app.Canvas):

    def __init__(self, title="Vispy Canvas"):

        app.Canvas.__init__(self, title=title)
        self.programs = {}
        self.controler = None

        gloo.set_viewport(0, 0, *self.physical_size)

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))


    def add_single_box(self, box_corner_positions=None):
        if box_corner_positions is None:
            box_corner_positions = np.array([[+0.9, +0.9],
                                             [-0.9, +0.9],
                                             [-0.9, -0.9],
                                             [+0.9, -0.9],
                                             [+0.9, +0.9]], dtype=np.float32)

        self.programs['box'] = LinesPlot(SINGLE_BOX_VERT_SHADER, SINGLE_BOX_FRAG_SHADER)
        self.programs['box']['a_position'] = box_corner_positions

    def add_multi_boxes(self, probe, box_corner_positions=None):
        box_indices = np.repeat(np.arange(0, self.probe.nb_channels, dtype=np.float32), repeats=5)
        box_positions = np.c_[
            np.repeat(self.probe.x.astype(np.float32), repeats=5),
            np.repeat(self.probe.y.astype(np.float32), repeats=5),
        ]
        if box_corner_positions is None:
            box_corner_positions = np.c_[
                np.tile(np.array([+1.0, -1.0, -1.0, +1.0, +1.0], dtype=np.float32), reps=self.nb_channels),
                np.tile(np.array([+1.0, +1.0, -1.0, -1.0, +1.0], dtype=np.float32), reps=self.nb_channels),
            ]
        # Define GLSL program.
        self.programs['box'] = LinesPlot(MULTI_BOX_VERT_SHADER, MULTI_BOX_FRAG_SHADER)
        self.programs['box']['a_box_index'] = box_indices
        self.programs['box']['a_box_position'] = box_positions
        self.programs['box']['a_corner_position'] = box_corner_positions
        self.programs['box']['u_x_min'] = self.probe.x_limits[0]
        self.programs['box']['u_x_max'] = self.probe.x_limits[1]
        self.programs['box']['u_y_min'] = self.probe.y_limits[0]
        self.programs['box']['u_y_max'] = self.probe.y_limits[1]
        self.programs['box']['u_d_scale'] = self.probe.minimum_interelectrode_distance

    def add_curve(self, name, plot_type, BOX_VERT_SHADER, BOX_FRAG_SHADER):
        if plot_type == 'lines':
            self.programs[name] = LinesPlot(BOX_VERT_SHADER, BOX_FRAG_SHADER)
        elif plot_type == 'points':
            self.programs[name] = ScatterPlot(BOX_VERT_SHADER, BOX_FRAG_SHADER)

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)
        return

    def get_colors(self, nb_templates, seed=42):
        np.random.seed(seed)
        return np.random.uniform(size=(nb_templates, 3), low=0.3, high=.9).astype(np.float32)

    def on_draw(self, event):

        _ = event
        gloo.clear()
        gloo.set_viewport(0, 0, *self.physical_size)
        for p in self.programs.values():
            p._draw()
        self.update()
        return

    def on_reception(self, data):
        self._on_reception(data)
        self.update()

    def set_value(self, dictionary):
        for key, value in dictionary.items():
            self._set_value(key, value)
        self.update()

    def highlight_selection(self, selection):
        self._highlight_selection(selection)
        self.update()
        return