from vispy import app, gloo
from vispy.util import keys
import numpy as np


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


class BoxPlot(LinesPlot):

    VERT_SHADER = """
    attribute vec2 a_position;
    void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
    """

    FRAG_SHADER = """
    // Fragment shader.
    void main() {
        gl_FragColor = vec4(0.5, 0.5, 0.5, 1.0);
    }
    """
    def __init__(self, box_corner_positions=None):
        LinesPlot.__init__(self, self.VERT_SHADER, self.FRAG_SHADER)
        if box_corner_positions is None:
            box_corner_positions = np.array([[+0.9, +0.9],
                                             [-0.9, +0.9],
                                             [-0.9, -0.9],
                                             [+0.9, -0.9],
                                             [+0.9, +0.9]], dtype=np.float32)
            self.set_corners(box_corner_positions)

    def set_corners(self, box_corner_positions):
        self.__setitem__('a_position', gloo.VertexBuffer(box_corner_positions))


class SingleLinePlot(LinesPlot):

    FRAG_SHADER = """
    varying float v_xpos;
    varying float v_index_selection;
    varying vec3 v_colors;
    varying float v_selection;

    // Fragment shader.
    void main() {
        gl_FragColor = vec4(v_colors, 1.0);
        if (v_selection == 0.0)
            discard;
        if (fract(v_index_selection) > 0.0 || (v_xpos) > 0.9)
            discard;
    }
    """

    VERT_SHADER = """
    attribute float a_values;
    attribute float a_selection;
    attribute vec3 a_colors;
    attribute float a_index_selection;
    attribute float a_index_xaxis;

    uniform vec2 u_scale;
    uniform float u_max_value;
    uniform float u_nb_points;

    varying vec3 v_colors;
    varying float v_selection;
    varying float v_index_selection;
    varying float v_xpos;

    void main() {
        float x = -0.9 + (1.8 * (a_index_xaxis / u_nb_points) * u_scale.x);
        float y = -0.9 + a_values/u_max_value;
        vec2 position = vec2(x, y);   
        gl_Position = vec4(position, 0.0, 1.0);
        v_colors = a_colors;
        v_index_selection = a_index_selection;
        v_xpos = position.x;
        v_selection = a_selection;
    }
    """

    def __init__(self, data=np.zeros((0, 0), dtype=np.float32)):
        LinesPlot.__init__(self, self.VERT_SHADER, self.FRAG_SHADER)
        self.zoom = np.array([1.0, 1.0], dtype=np.float32)
        self.selection = []
        self.set_data(data)

    @property
    def nb_data(self):
        return self.data.shape[0]

    @property
    def nb_points(self):
        return self.data.shape[1]

    @property
    def initialized(self):
        if self.nb_points > 0:
            return True
        else:
            return False

    @property
    def max_y_axis(self):
        if self.initialized:
            return np.max(self.data)
        else:
            return 0

    def _generate_selection(self, selection=[]):

        if selection is not None and self.selection != selection:
            self.selection = selection
        data = np.zeros(self.nb_data, dtype=np.int32)
        data[self.selection] = 1
        return data

    def set_attribute(self, attribute, data):
        #set_attr(self, attribute, data)

        if attribute in ['a_selection', 'a_index_selection']:
            data = np.repeat(data, repeats=self.nb_points).astype(np.float32)
        elif attribute in ['a_colors']:
            data = np.repeat(data, repeats=self.nb_points, axis=0).astype(np.float32)
        elif attribute in ['a_values']:
            data = data.ravel().astype(np.float32)
        elif attribute in ['a_index_xaxis']:
            data = np.tile(data, reps=self.nb_data).astype(np.float32)

        if attribute in ['a_selection', 'a_index_selection', 'a_colors', 'a_values', 'a_index_xaxis']:
            self.__setitem__(attribute, gloo.VertexBuffer(data))
        else:
            self.__setitem__(attribute, data)

    def set_zoom_y_axis(self, factor=1):
        self.zoom[1] = factor
        self.set_attribute('u_scale', self.zoom)

    def set_data(self, data, colors=None):

        self.data = data
        if colors is None:
            colors = np.zeros((self.nb_data, 3), dtype=np.float32)

        self.set_attribute('a_values', data)
        self.set_attribute('a_colors', colors)
        self.set_attribute('a_index_xaxis', np.arange(self.nb_points))
        self.set_attribute('a_index_selection', np.arange(self.nb_data))
        self.set_attribute('u_scale', self.zoom)
        self.set_attribute('u_max_value', self.max_y_axis)
        self.set_attribute('u_nb_points', self.nb_points)
        self.set_selection(None)

    def set_selection(self, selection):
        selection = self._generate_selection(selection)   
        self.set_attribute('a_selection', selection)
