import numpy as np

from vispy import app, gloo
from vispy.util import keys

from circusort.io.probe import load_probe
from circusort.io.template import load_template_from_dict

BOUNDARY_VERT_SHADER = """

// Coordinates of the position of the box.
attribute vec2 a_pos_probe;
// Coordinates of the position of the corner.
attribute vec2 a_corner_position;
// Uniform variables used to transform the subplots.
uniform float u_x_min;
uniform float u_x_max;
uniform float u_y_min;
uniform float u_y_max;
uniform float u_d_scale;

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
    vec2 b = vec2(-1.0 + 2.0 * (a_pos_probe.x - u_x_min) / w,
                    -1.0 + 2.0 * (a_pos_probe.y - u_y_min) / h);
    // Apply the transformation.
    gl_Position = vec4(a * p + b, 0.0, 1.0);
}
"""

TEST_VERT_SHADER = """
attribute vec2 a_pos;
void main(){
gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

BOUNDARY_FRAG_SHADER = """
// Fragment shader.
void main() {
    gl_FragColor = vec4(0.25, 0.25, 0.25, 1.0);
}
"""

TEST_FRAG_SHADER = """
// Fragment shader.
void main() {
    gl_FragColor = vec4(0.1, 1.0, 0.1, 1.0);
}
"""

class MEACanvas(app.Canvas):

    def __init__(self, probe_path=None, params=None):
        app.Canvas.__init__(self, title="Vispy canvas2", keys="interactive")

        self.probe = load_probe(probe_path)
        # self.channels = params['channels']
        self.nb_channels = 9

        # TODO Add method to probe file to extract minimum coordinates without the interelectrode dist
        x_min, x_max = self.probe.x_limits[0] + self.probe.minimum_interelectrode_distance,\
                       self.probe.x_limits[1] - self.probe.minimum_interelectrode_distance
        y_min, y_max = self.probe.y_limits[0] + self.probe.minimum_interelectrode_distance, \
                       self.probe.y_limits[1] - self.probe.minimum_interelectrode_distance

        probe_corner = np.array([[x_max, y_max],
                                 [x_min, y_max],
                                 [x_min, y_min],
                                 [x_max, y_min],
                                 [x_max, y_max]], dtype=np.float32)

        corner_bound_positions = np.array([[+1.0, +1.0],
                                           [-1.0, +1.0],
                                           [-1.0, -1.0],
                                           [+1.0, -1.0],
                                           [+1.0, +1.0]], dtype=np.float32)

        # Define GLSL program.
        self._boundary_program = gloo.Program(vert=BOUNDARY_VERT_SHADER, frag=BOUNDARY_FRAG_SHADER)
        self._boundary_program['a_pos_probe'] = probe_corner
        self._boundary_program['a_corner_position'] = corner_bound_positions
        self._boundary_program['u_x_min'] = self.probe.x_limits[0]
        self._boundary_program['u_x_max'] = self.probe.x_limits[1]
        self._boundary_program['u_y_min'] = self.probe.y_limits[0]
        self._boundary_program['u_y_max'] = self.probe.y_limits[1]
        self._boundary_program['u_d_scale'] = self.probe.minimum_interelectrode_distance

        test = np.array([[0.5, 0.5],
                         [0.5, -0.5],
                         [-0.5, -0.5],
                         [-0.5, 0.5]], dtype=np.float32)
        self._test_program = gloo.Program(vert=TEST_VERT_SHADER, frag=TEST_FRAG_SHADER)
        self._test_program['a_pos'] = test

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
        self._boundary_program.draw('line_strip')
        self._test_program.draw('line_strip')
        return
