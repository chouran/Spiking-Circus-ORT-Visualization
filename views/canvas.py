import utils.widgets as wid
import numpy as np

from vispy import app, gloo
from vispy.util import keys

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


class ViewCanvas(app.Canvas):

    def __init__(self, title="Vispy Canvas", show_box=False):

        app.Canvas.__init__(self, title=title)

        self.programs = {}
        self.controler = None

        if show_box:
            box_corner_positions = np.array([[+0.9, +0.9],
                                         [-0.9, +0.9],
                                         [-0.9, -0.9],
                                         [+0.9, -0.9],
                                         [+0.9, +0.9]], dtype=np.float32)

            self.programs['box'] = gloo.Program(vert=BOX_VERT_SHADER, frag=BOX_FRAG_SHADER)
            self.programs['box']['a_position'] = box_corner_positions

        gloo.set_viewport(0, 0, *self.physical_size)

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    def add_curve(self, name, BOX_FRAG_SHADER, BOX_VERT_SHADER):
        self.programs[name] = gloo.Program(vert=BOX_VERT_SHADER, frag=BOX_FRAG_SHADER)

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)
        return

    def on_draw(self, event):

        _ = event
        gloo.clear()
        gloo.set_viewport(0, 0, *self.physical_size)
        for p in self.programs.values():
            p.draw('line_strip')
        return

    def on_reception(self, data):
        self._on_reception(self, data)
        self.update()

    def set_value(self, key, value):
        self._set_value(self, key, value)
        self.update()

    def highlight_selection(self, selection):
        self._highlight_selection(selection)
        self.update()
        return