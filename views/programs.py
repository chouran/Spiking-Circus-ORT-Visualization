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
    varying float v_xaxis;

    void main() {
        float x = -0.9 + (1.8 * (a_index_xaxis / u_nb_points) * u_scale.x);
        float y = -0.9 + a_values/u_max_value;
        vec2 position = vec2(x, y);   
        gl_Position = vec4(position, 0.0, 1.0);
        v_color = a_colors;
        v_index_cell = a_index_selection;
        v_xpos = position.x;
        v_selection = a_selection;
    }
    """

    def __init__(self, nb_points=0, nb_templates=0):
        self.selection = []
        LinesPlot.__init__(self, self.FRAG_SHADER, self.VERT_SHADER)
        LinesPlot.set('a_values', np.zeros((nb_templates, nb_points), dtype=np.float32))

    def _highlight_selection(self, selection, nb_templates):
        self.selection = [0] * self.nb_templates
        for i in selection:
            self.list_selected_cells[i] = 1
        self.set('a_selection', self.selection) 
        return

    def set_attribute(self, attribute, data):
        self[attribute] = data
        self.selection = np.repeat(self.list_selected_cells, repeats=self.nb_points).astype(np.float32)

        self.selected_isi_vector = np.repeat(self.list_selected_cells, repeats=self.nb_points).astype(
                np.float32)

        # self.index_x = np.tile(np.arange(self.nb_points), reps=self.nb_templates).astype(np.float32)
        # self.index_cell = np.repeat(np.arange(self.nb_templates), repeats=self.nb_points).astype(np.float32)
        # self.color_isi = np.repeat(self.get_colors(self.nb_templates), repeats=self.nb_points, axis=0)

        # self.programs['isis']['a_isi_value'] = self.isi_vector.ravel()
        # self.programs['isis']['a_selected_cell'] = self.selected_isi_vector
        # self.programs['isis']['a_color'] = self.color_isi
        # self.programs['isis']['a_index_x'] = self.index_x
        # self.programs['isis']['a_index_cell'] = self.index_cell
        # self.programs['isis']['u_scale'] = self.u_scale
        # self.programs['isis']['u_nb_points'] = self.nb_points
        # self.programs['isis']['u_max_value'] = self.max_isi
