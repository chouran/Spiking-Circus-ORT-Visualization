from vispy import app, gloo
from vispy.util import keys
import re

class ORTPlot(gloo.Program):

    def __init__(self, vert, frag):
        self.vert = vert
        self.frag = frag
        gloo.Program.__init__(self, vert=vert, frag=frag)

    # @property
    # def uniforms(self)
    #     re.compile('\w+)\s+(\w+)\s+(\w+)\s*;')



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



class SingleLinePlot(LinesPlot):

    FRAG_SHADER = """
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

    VERT_SHADER = """
    attribute float a_value;
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
        float y = -0.9 + a_value/u_max_value;
        vec2 position = vec2(x, y);   
        gl_Position = vec4(position, 0.0, 1.0);
        v_color = a_color;
        v_index_cell = a_index_cell;
        v_pos_x = position.x;
        v_selected_cell = a_selected_cell;
    }
    """

    def __init__(self, nb_channels):
        LinesPlot.__init__(self, self.FRAG_SHADER, self.VERT_SHADER)

