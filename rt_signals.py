import vispy
from vispy import app
from vispy import gloo
app.use_app('pyqt5')
import numpy as np
import math

#Parameters
nrows = 3
ncols = 3
nb_signals = nrows * ncols
nb_samples = 1000
amplitudes = .1 + .2 * np.random.rand(nb_signals, 1).astype(np.float32)
amp_range = (0.1, 0.3)
time_range = (0, 2*0.9/ncols)

# Generate the signals as a (m, n) array.
y = amplitudes * np.random.randn(nb_signals, nb_samples).astype(np.float32)

# Color of each vertex (TODO: make it more efficient by using a GLSL-based
# color map and the index).
color = np.repeat(np.random.uniform(size=(nb_signals, 3), low=.5, high=.9),
                  nb_samples, axis=0).astype(np.float32)

# Signal 2D index of each vertex (row and col) and x-index (sample index
# within each signal).
index = np.c_[np.repeat(np.repeat(np.arange(ncols), nrows), nb_samples),
              np.repeat(np.tile(np.arange(nrows), ncols), nb_samples),
              np.tile(np.arange(nb_samples), nb_signals)].astype(np.float32)

#Boxes Parameters
box_index = np.c_[np.repeat(np.repeat(np.arange(ncols), nrows), 5),
              np.repeat(np.tile(np.arange(nrows), ncols), 5)].astype(np.float32)

box_number = np.repeat(np.arange(0, nb_signals, dtype=np.float32), repeats=5)

corner_positions = np.c_[
    np.tile(np.array([+1.0, -1.0, -1.0, +1.0, +1.0], dtype=np.float32), reps=nb_signals),
    np.tile(np.array([+1.0, +1.0, -1.0, -1.0, +1.0], dtype=np.float32), reps=nb_signals),
    ]

#Thresholds parameters
th_value = 0.0
th_position_ini = np.c_[
    np.tile(np.array([+1.0, -1.0], dtype=np.float32), reps=nb_signals),
    np.tile(np.array([th_value, th_value], dtype=np.float32), reps=nb_signals),]

th_index = np.c_[np.repeat(np.repeat(np.arange(ncols), nrows), 2),
              np.repeat(np.tile(np.arange(nrows), ncols), 2)].astype(np.float32)

#Vertex Shaders

SIGNAL_VERT_SHADER = """
#version 120
// y coordinate of the position.
attribute float a_position;
// row, col, and time index.
attribute vec3 a_index;
varying vec3 v_index;
// 2D scaling factor (zooming).
uniform vec2 u_scale;
// Size of the table.
uniform vec2 u_size;
// Number of samples per signal.
uniform float u_n;
// Color.
attribute vec3 a_color;
varying vec4 v_color;
// Varying variables used for clipping in the fragment shader.
varying vec2 v_position;
varying vec4 v_ab;
void main() {
    float nrows = u_size.x;
    float ncols = u_size.y;
    // Compute the x coordinate from the time index.
    float x = -1 + 2*a_index.z / (u_n-1);
    vec2 position = vec2(x - (1 - 1 / u_scale.x), a_position);
    // Find the affine transformation for the subplots.
    vec2 a = vec2(1./ncols, 1./nrows)*.9;
    vec2 b = vec2(-1 + 2*(a_index.x+.5) / ncols,
                  -1 + 2*(a_index.y+.5) / nrows);
    // Apply the static subplot transformation + scaling.
    gl_Position = vec4(a*u_scale*position+b, 0.0, 1.0);
    v_color = vec4(a_color, 1.);
    v_index = a_index;
    // For clipping test in the fragment shader.
    v_position = gl_Position.xy;
    v_ab = vec4(a, b);
}
"""


BOX_VERT_SHADER = """
// Index of the box.
attribute vec2 a_box_index;

// Box number
attribute float a_box_number;

// Coordinates of the position of the corner.
attribute vec2 a_corner_position;

// 2D scaling factor (zooming).
uniform vec2 u_scale;
// Size of the table.
uniform vec2 u_size;

// Varying variable used for clipping in the fragment shader.
varying float v_box_nb;

void main() {
    float nrows = u_size.x;
    float ncols = u_size.y;

    vec2 position = a_corner_position;
    // Find the affine transformation for the subplots.
    vec2 a = vec2(1./ncols, 1./nrows)*.9;
    vec2 b = vec2(-1 + 2*(a_box_index.x+.5) / ncols,
                  -1 + 2*(a_box_index.y+.5) / nrows);
    // Apply the static subplot transformation + scaling.
    gl_Position = vec4(a*u_scale*position+b, 0.0, 1.0);
    v_box_nb = a_box_number;
}
"""

THRESHOLD_VERT_SHADER = """
// Index of the box.
attribute vec2 a_threshold_index;

// Coordinates of the position of the corner.
attribute vec2 a_threshold_position;

// 2D scaling factor (zooming).
uniform vec2 u_scale;
// Size of the table.
uniform vec2 u_size;

// Varying variable used for clipping in the fragment shader.
varying vec2 v_index;

// Varying variables used for clipping in the fragment shader.
varying vec2 v_position;
varying vec4 v_ab;

void main() {
    float nrows = u_size.x;
    float ncols = u_size.y;

    vec2 position = a_threshold_position;
    // Find the affine transformation for the subplots.
    vec2 a = vec2(1./ncols, 1./nrows)*.9;
    vec2 b = vec2(-1 + 2*(a_threshold_index.x+.5) / ncols,
                  -1 + 2*(a_threshold_index.y+.5) / nrows);
    // Apply the static subplot transformation + scaling.
    gl_Position = vec4(a*u_scale*position+b, 0.0, 1.0);
    v_index = a_threshold_index;
    // For clipping test in the fragment shader.
    v_position = gl_Position.xy;
    v_ab = vec4(a, b);
}
"""

#Pixel Shaders
SIGNAL_FRAG_SHADER = """
#version 120
varying vec4 v_color;
varying vec3 v_index;
varying vec2 v_position;
varying vec4 v_ab;
void main() {
    gl_FragColor = v_color;
    // Discard the fragments between the signals (emulate glMultiDrawArrays).
    if ((fract(v_index.x) > 0.) || (fract(v_index.y) > 0.))
        discard;
    // Clipping test.
    vec2 test = abs((v_position.xy-v_ab.zw)/v_ab.xy);
    if ((test.x > 1) || (test.y > 1))
        discard;
}
"""


BOX_FRAG_SHADER = """
// Varying variable.
varying float v_box_nb;

void main() {
    gl_FragColor = vec4(0.25, 0.25, 0.25, 1.0);
    // Discard the fragments between the box (emulate glMultiDrawArrays).
    if (fract(v_box_nb) > 0.0)
        discard;    
}
"""

THRESHOLD_FRAG_SHADER = """
void main() {
    gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0); 
}
"""

class SignalCanvas(app.Canvas):
    #gloo.context.FakeCanvas

    def __init__(self):
        app.Canvas.__init__(self, title='Use your wheel to zoom!', show=False,
                            keys='interactive')

        #Signal
        self.program = gloo.Program(SIGNAL_VERT_SHADER, SIGNAL_FRAG_SHADER)
        self.program['a_position'] = y.reshape(-1, 1)
        self.program['a_color'] = color
        self.program['a_index'] = index
        self.program['u_scale'] = (1., 1.)
        self.program['u_size'] = (nrows, ncols)
        self.program['u_n'] = nb_samples

        #Box
        self.program_box = gloo.Program(vert=BOX_VERT_SHADER, frag=BOX_FRAG_SHADER)
        self.program_box['a_box_index'] = box_index
        self.program_box['a_box_number'] = box_number
        self.program_box['a_corner_position'] = corner_positions
        self.program_box['u_size'] = (nrows, ncols)
        self.program_box['u_scale'] = (1., 1.)

        #Threshold
        self.program_th = gloo.Program(vert=THRESHOLD_VERT_SHADER,
                                       frag=THRESHOLD_FRAG_SHADER)
        self.program_th['a_threshold_index'] = th_index
        self.program_th['a_threshold_position'] = th_position_ini
        self.program_th['u_size'] = (nrows, ncols)
        self.program_th['u_scale'] = (1., 1.)

        gloo.set_viewport(0, 0, *self.physical_size)

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))
        self.show()

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_mouse_wheel(self, event):
        dx = np.sign(event.delta[1]) * .05
        scale_x, scale_y = self.program['u_scale']
        scale_x_new, scale_y_new = (scale_x * math.exp(2.5 * dx),
                                    scale_y * math.exp(0.0 * dx))
        self.program['u_scale'] = (max(1, scale_x_new), max(1, scale_y_new))
        self.update()

    def on_timer(self, event):
        """Add some data at the end of each signal (real-time signals)."""
        k = 10
        y[:, :-k] = y[:, k:]
        y[:, -k:] = amplitudes * np.random.randn(nb_signals, k)

        self.program['a_position'].set_data(y.ravel().astype(np.float32))
        self.update()

    def update_threshold(self, th):
        th_position = np.c_[
                    np.tile(np.array([+1.0, -1.0], dtype=np.float32), reps=nb_signals),
                    np.tile(np.array([th, th], dtype=np.float32), reps=nb_signals)]
        self.program_th['a_threshold_position'] = th_position
        self.update()

    def on_draw(self, event):
        gloo.clear()
        self.program.draw('line_strip')
        self.program_box.draw('line_strip')
        self.program_th.draw('lines')

#c = SignalCanvas()
#app.run()

