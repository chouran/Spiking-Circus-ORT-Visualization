import utils.widgets as wid
import numpy as np

from vispy import app, gloo
from vispy.util import keys


class ViewCanvas(app.Canvas):

    def __init__(self, title="Vispy Canvas"):

        app.Canvas.__init__(self, title=title)

        self._programs = []
        gloo.set_viewport(0, 0, *self.physical_size)

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    def add_curve(self, gloo_program):
        self._programs += [gloo_program]

    @staticmethod
    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)
        return

    def on_draw(self, event):

        _ = event
        gloo.clear()
        gloo.set_viewport(0, 0, *self.physical_size)
        for p in self._programs:
            p.draw('line_strip')
    
        return

    def on_reception(self, data):
        self._on_reception(self, data)
        self.update()

    def set_value(self, key, value):
        self._set_value(self, key, value)
        self.update()