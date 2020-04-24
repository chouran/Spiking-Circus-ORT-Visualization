import numpy as np

from vispy import app, gloo
from vispy.util import keys

from circusort.io.probe import load_probe

SIGNAL_VERT_SHADER = """
// Index of the signal.
attribute float a_signal_index;
// Coordinates of the position of the signal.
attribute vec2 a_signal_position;
// Value of the signal.
attribute float a_signal_value;
// Color of the signal.
//attribute vec3 a_signal_color;
// Index of the sample of the signal.
attribute float a_sample_index;
// Number of samples per signal.
uniform float u_nb_samples_per_signal;
// Uniform variables used to transform the subplots.
uniform float u_x_min;
uniform float u_x_max;
uniform float u_y_min;
uniform float u_y_max;
uniform float u_d_scale;
uniform float u_t_scale;
uniform float u_v_scale;
// Varying variables used for clipping in the fragment shader.
varying float v_index;
varying vec4 v_color;
varying vec2 v_position;

// values of the MADs, for spike identification
attribute float a_spike_threshold;
varying float v_spike_threshold;
uniform float see_spikes;
varying float v_see_spikes;

// Vertex shader.
void main() {
    // Compute the x coordinate from the sample index.
    float x = +1.0 + 2.0 * u_t_scale * (-1.0 + (a_sample_index / (u_nb_samples_per_signal - 1.0)));
    // Compute the y coordinate from the signal value.
    float y =  a_signal_value / u_v_scale;
    // Compute the position.
    vec2 p = vec2(x, y);
    // Affine transformation for the subplots.
    float w = u_x_max - u_x_min;
    float h = u_y_max - u_y_min;
    vec2 a = vec2(1.0 / (1.0 + w / u_d_scale), 1.0 / (1.0 + h / u_d_scale));
    vec2 b = vec2(-1.0 + 2.0 * (a_signal_position.x - u_x_min) / w, -1.0 + 2.0 * (a_signal_position.y - u_y_min) / h);
    vec2 p_ = a * p + b;
    // Compute GL position.
    gl_Position = vec4(p_, 0.0, 1.0);
    // TODO remove the following;
    v_index = a_signal_index;
    //v_color = vec4(a_signal_color, 1.0);
    v_position = p;
    
    v_spike_threshold = float(a_spike_threshold/u_v_scale);
    v_see_spikes = float(see_spikes);
}
"""

MADS_VERT_SHADER = """
// Index of the MADs.
attribute float a_mads_index;
// Coordinates of the position of the MADs.
attribute vec2 a_mads_position;
// Value of the MADs.
attribute float a_mads_value;
// Color of the MADs.
attribute vec3 a_mads_color;
// Index of the sample of the MADs.
attribute float a_sample_index;
// Number of samples per signal.
uniform float u_nb_samples_per_signal;
// Uniform variables used to transform the subplots.
uniform float u_x_min;
uniform float u_x_max;
uniform float u_y_min;
uniform float u_y_max;
uniform float u_d_scale;
uniform float u_t_scale;
uniform float u_v_scale;
uniform bool display;
// Varying variables used for clipping in the fragment shader.
varying vec4 v_color;
varying float v_index;
varying vec2 v_position;
// Vertex shader.
void main() {
    // Compute the x coordinate from the sample index.
    float x = +1.0 + 2.0 * u_t_scale * (-1.0 + (a_sample_index / (u_nb_samples_per_signal - 1.0)));
    // Compute the y coordinate from the signal value.
    float y =  a_mads_value / u_v_scale;
    // Compute the position.
    vec2 p = vec2(x, y);
    // Affine transformation for the subplots.
    float w = u_x_max - u_x_min;
    float h = u_y_max - u_y_min;
    vec2 a = vec2(1.0 / (1.0 + w / u_d_scale), 1.0 / (1.0 + h / u_d_scale));
    vec2 b = vec2(-1.0 + 2.0 * (a_mads_position.x - u_x_min) / w, -1.0 + 2.0 * (a_mads_position.y - u_y_min) / h);
    vec2 p_ = a * p + b;
    // Compute GL position.
    gl_Position = vec4(p_, 0.0, 1.0);
    // Define varying variables.
    if (display == true)
        v_color = vec4(a_mads_color, 1.0);
    else
        v_color = vec4(0.0, 0.0, 0.0, 0.0);
    v_index = a_mads_index;
    v_position = p;
}
"""

PEAKS_VERT_SHADER = """
// Index of the MADs.
attribute float a_peaks_index;
// Coordinates of the position of the MADs.
attribute vec2 a_peaks_position;
// Value of the MADs.
attribute float a_peaks_sizes;
// Color of the MADs.
attribute vec3 a_peaks_color;
// Index of the sample of the MADs.
attribute float a_sample_index;
// Number of samples per signal.
uniform float u_nb_samples_per_signal;
// Uniform variables used to transform the subplots.
uniform float u_x_min;
uniform float u_x_max;
uniform float u_y_min;
uniform float u_y_max;
uniform float u_d_scale;
uniform float u_t_scale;
uniform float u_v_scale;
uniform bool display;
// Varying variables used for clipping in the fragment shader.
varying vec4 v_color;
varying float v_index;
varying vec2 v_position;
// Vertex shader.
void main() {
    v_radius = a_peaks_sizes;
    v_linewidth = 1.0;
    v_antialias = 1.0;
    // Compute the x coordinate from the sample index.
    float x = +1.0 + 2.0 * u_t_scale * (-1.0 + (a_sample_index / (u_nb_samples_per_signal - 1.0)));
    // Compute the y coordinate from the signal value.

    // Compute the position.
    vec2 p = a_peaks_position; //vec2(x, y);
    // Affine transformation for the subplots.
    float w = u_x_max - u_x_min;
    float h = u_y_max - u_y_min;
    vec2 a = vec2(1.0 / (1.0 + w / u_d_scale), 1.0 / (1.0 + h / u_d_scale));
    vec2 b = vec2(-1.0 + 2.0 * (a_peaks_position.x - u_x_min) / w, -1.0 + 2.0 * (a_peaks_position.y - u_y_min) / h);
    vec2 p_ = a * p + b;
    // Compute GL position.
    gl_Position = vec4(p_, 0.0, 1.0);
    // Define varying variables.
    if (display == true)
        v_color = vec4(a_peaks_color, 1.0);
    else
        v_color = vec4(0.0, 0.0, 0.0, 0.0);
    gl_PointSize = 2.0*(v_radius + v_linewidth + 1.5*v_antialias);
}
"""

BOX_VERT_SHADER = """
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



SIGNAL_FRAG_SHADER = """
// Varying variables.
varying vec4 v_color;
varying float v_index;
varying vec2 v_position;

varying float v_see_spikes;
varying float v_spike_threshold;

// Fragment shader.
void main() {
    //gl_FragColor = v_color;
    
    if (v_position.y > v_spike_threshold && v_see_spikes == 1.0)
        gl_FragColor = vec4(0.9, 0.0, 0.0, 1.0);
    else
        gl_FragColor = vec4(0.9, 0.9, 0.9, 1.0);
    
    // Discard the fragments between the signals (emulate glMultiDrawArrays).
    if (fract(v_index) > 0.0)
        discard;
    // Clipping test.
    if ((abs(v_position.x) > 1.0) || (abs(v_position.y) > 1))
        discard;
}
"""

MADS_FRAG_SHADER = """
// Varying variables.
varying vec4 v_color;
varying float v_index;
varying vec2 v_position;
// Fragment shader.
void main() {
    gl_FragColor = v_color;
    // Discard the fragments between the MADs (emulate glMultiDrawArrays).
    if (fract(v_index) > 0.0)
        discard;
    // Clipping test.
    if ((abs(v_position.x) > 1.0) || (abs(v_position.y) > 1))
        discard;
}
"""

PEAKS_FRAG_SHADER = """
// Varying variables.
varying vec4 v_color;
varying float v_index;
varying vec2 v_position;
// Fragment shader.
void main() {
    gl_FragColor = v_color;
    // Discard the fragments between the MADs (emulate glMultiDrawArrays).
    if (fract(v_index) > 0.0)
        discard;
    // Clipping test.
    if ((abs(v_position.x) > 1.0) || (abs(v_position.y) > 1))
        discard;
}
"""

BOX_FRAG_SHADER = """
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


class TraceCanvas(app.Canvas):

    def __init__(self, probe_path=None, params=None):

        app.Canvas.__init__(self, title="Vispy canvas", keys="interactive")

        self.probe = load_probe(probe_path)
        
        nb_buffers_per_signal = int(np.ceil((params['time']['max'] * 1e-3) * params['sampling_rate']
                                            / float(params['nb_samples'])))
        self._time_max = (float(nb_buffers_per_signal * params['nb_samples']) / params['sampling_rate']) * 1e+3
        self._time_min = params['time']['min']
        self.mad_factor = params['mads']['init']
        self.channels = params['channels']

        # Signals.

        # Number of signals.
        self.nb_signals = self.probe.nb_channels
        # Number of samples per buffer.
        self._nb_samples_per_buffer = params['nb_samples']
        # Number of samples per signal.
        nb_samples_per_signal = nb_buffers_per_signal * self._nb_samples_per_buffer
        # Generate the signal values.
        self._signal_values = np.zeros((self.nb_signals, nb_samples_per_signal), dtype=np.float32)
        # Color of each vertex.
        # TODO: make it more efficient by using a GLSL-based color map and the index.
        signal_colors = 0.75 * np.ones((self.nb_signals, 3), dtype=np.float32)
        signal_colors = np.repeat(signal_colors, repeats=nb_samples_per_signal, axis=0)
        signal_indices = np.repeat(np.arange(0, self.nb_signals, dtype=np.float32), repeats=nb_samples_per_signal)
        signal_positions = np.c_[
            np.repeat(self.probe.x.astype(np.float32), repeats=nb_samples_per_signal),
            np.repeat(self.probe.y.astype(np.float32), repeats=nb_samples_per_signal),
        ]
        sample_indices = np.tile(np.arange(0, nb_samples_per_signal, dtype=np.float32), reps=self.nb_signals)

        # Mads with the appropriate shape
        mads_thresholds = np.zeros((nb_samples_per_signal * self.nb_signals,), dtype=np.float32)

        # Define GLSL program.
        self._signal_program = gloo.Program(vert=SIGNAL_VERT_SHADER, frag=SIGNAL_FRAG_SHADER)
        self._signal_program['a_signal_index'] = gloo.VertexBuffer(signal_indices)
        self._signal_program['a_signal_position'] = gloo.VertexBuffer(signal_positions)
        self._signal_program['a_signal_value'] = gloo.VertexBuffer(self._signal_values.reshape(-1, 1))
        self._signal_program['a_spike_threshold'] = mads_thresholds
        self._signal_program['see_spikes'] = 1.0
        self._signal_program['a_sample_index'] = gloo.VertexBuffer(sample_indices)
        self._signal_program['u_nb_samples_per_signal'] = nb_samples_per_signal
        self._signal_program['u_x_min'] = self.probe.x_limits[0]
        self._signal_program['u_x_max'] = self.probe.x_limits[1]
        self._signal_program['u_y_min'] = self.probe.y_limits[0]
        self._signal_program['u_y_max'] = self.probe.y_limits[1]
        self._signal_program['u_d_scale'] = self.probe.minimum_interelectrode_distance
        self._signal_program['u_t_scale'] = self._time_max / params['time']['init']
        self._signal_program['u_v_scale'] = params['voltage']['init']

        # MADs.

        # Generate the MADs values.
        mads_indices = np.arange(0, self.nb_signals, dtype=np.float32)
        mads_indices = np.repeat(mads_indices, repeats=2 * (nb_buffers_per_signal + 1))
        mads_positions = np.c_[
            np.repeat(self.probe.x.astype(np.float32), repeats=2 * (nb_buffers_per_signal + 1)),
            np.repeat(self.probe.y.astype(np.float32), repeats=2 * (nb_buffers_per_signal + 1)),
        ]
        self._mads_values = np.zeros((self.nb_signals, 2 * (nb_buffers_per_signal + 1)), dtype=np.float32)
        mads_colors = np.array([0.75, 0.0, 0.0], dtype=np.float32)
        mads_colors = np.tile(mads_colors, reps=(self.nb_signals, 1))
        mads_colors = np.repeat(mads_colors, repeats=2 * (nb_buffers_per_signal + 1), axis=0)
        sample_indices = np.arange(0, nb_buffers_per_signal + 1, dtype=np.float32)
        sample_indices = np.repeat(sample_indices, repeats=2)
        sample_indices = self._nb_samples_per_buffer * sample_indices
        sample_indices = np.tile(sample_indices, reps=self.nb_signals)

        # Define GLSL program.
        self._mads_program = gloo.Program(vert=MADS_VERT_SHADER, frag=MADS_FRAG_SHADER)
        self._mads_program['a_mads_index'] = gloo.VertexBuffer(mads_indices)
        self._mads_program['a_mads_position'] = gloo.VertexBuffer(mads_positions)
        self._mads_program['a_mads_value'] = gloo.VertexBuffer(self._mads_values.reshape(-1, 1))
        self._mads_program['a_mads_color'] = gloo.VertexBuffer(mads_colors)
        self._mads_program['a_sample_index'] = gloo.VertexBuffer(sample_indices)
        self._mads_program['u_nb_samples_per_signal'] = nb_samples_per_signal
        self._mads_program['u_x_min'] = self.probe.x_limits[0]
        self._mads_program['u_x_max'] = self.probe.x_limits[1]
        self._mads_program['u_y_min'] = self.probe.y_limits[0]
        self._mads_program['u_y_max'] = self.probe.y_limits[1]
        self._mads_program['u_d_scale'] = self.probe.minimum_interelectrode_distance
        self._mads_program['u_t_scale'] = self._time_max / params['time']['init']
        self._mads_program['u_v_scale'] = params['voltage']['init']
        self._mads_program['display'] = False

        # Peaks.
        peaks_positions = np.zeros((0, 2), dtype=np.float32)
        peaks_sizes = 10 * self.pixel_scale * np.ones(0, dtype=np.float32)
        peaks_colors = np.array([0.75, 0.0, 0.0], dtype=np.float32)
        peaks_colors = np.tile(peaks_colors, reps=(self.nb_signals, 1))
        peaks_colors = np.repeat(peaks_colors, repeats=2 * (nb_buffers_per_signal + 1), axis=0)
        print(peaks_colors.shape)

        self._peaks_program = gloo.Program(vert=PEAKS_VERT_SHADER, frag=PEAKS_FRAG_SHADER)
        self._peaks_program['a_peaks_position'] = gloo.VertexBuffer(peaks_positions)
        self._peaks_program['a_peaks_sizes'] = gloo.VertexBuffer(peaks_sizes)
        self._peaks_program['a_peaks_color'] = gloo.VertexBuffer(peaks_colors)
        self._peaks_program['u_x_min'] = self.probe.x_limits[0]
        self._peaks_program['u_x_max'] = self.probe.x_limits[1]
        self._peaks_program['u_y_min'] = self.probe.y_limits[0]
        self._peaks_program['u_y_max'] = self.probe.y_limits[1]
        self._peaks_program['u_d_scale'] = self.probe.minimum_interelectrode_distance
        self._peaks_program['u_t_scale'] = self._time_max / params['time']['init']
        self._peaks_program['display'] = True

        # Boxes.

        box_indices = np.repeat(np.arange(0, self.nb_signals, dtype=np.float32), repeats=5)
        box_positions = np.c_[
            np.repeat(self.probe.x.astype(np.float32), repeats=5),
            np.repeat(self.probe.y.astype(np.float32), repeats=5),
        ]
        corner_positions = np.c_[
            np.tile(np.array([+1.0, -1.0, -1.0, +1.0, +1.0], dtype=np.float32), reps=self.nb_signals),
            np.tile(np.array([+1.0, +1.0, -1.0, -1.0, +1.0], dtype=np.float32), reps=self.nb_signals),
        ]
        # Define GLSL program.
        self._box_program = gloo.Program(vert=BOX_VERT_SHADER, frag=BOX_FRAG_SHADER)
        self._box_program['a_box_index'] = gloo.VertexBuffer(box_indices)
        self._box_program['a_box_position'] = gloo.VertexBuffer(box_positions)
        self._box_program['a_corner_position'] = gloo.VertexBuffer(corner_positions)
        self._box_program['u_x_min'] = self.probe.x_limits[0]
        self._box_program['u_x_max'] = self.probe.x_limits[1]
        self._box_program['u_y_min'] = self.probe.y_limits[0]
        self._box_program['u_y_max'] = self.probe.y_limits[1]
        self._box_program['u_d_scale'] = self.probe.minimum_interelectrode_distance

        # Final details.

        gloo.set_viewport(0, 0, *self.physical_size)

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    @staticmethod
    def on_resize(event):

        gloo.set_viewport(0, 0, *event.physical_size)

        return

    def on_mouse_wheel(self, event):
    
        modifiers = event.modifiers

        if keys.CONTROL in modifiers:
            dx = np.sign(event.delta[1]) * 0.01
            v_scale = self._signal_program['u_v_scale']
            v_scale_new = v_scale * np.exp(dx)
            self._signal_program['u_v_scale'] = v_scale_new
            self._mads_program['u_v_scale'] = v_scale_new
        elif keys.SHIFT in modifiers:
            time_ref = self._time_max
            dx = np.sign(event.delta[1]) * 0.01
            t_scale = self._signal_program['u_t_scale']
            t_scale_new = t_scale * np.exp(dx)
            t_scale_new = max(t_scale_new, time_ref / self._time_max)
            t_scale_new = min(t_scale_new, time_ref / self._time_min)
            self._signal_program['u_t_scale'] = t_scale_new
            self._mads_program['u_t_scale'] = t_scale_new
        else:
            dx = np.sign(event.delta[1]) * 0.01
            x_min_new = self._signal_program['u_x_min'] * np.exp(dx)
            x_max_new = self._signal_program['u_x_max'] * np.exp(dx)
            self._signal_program['u_x_min'] = x_min_new
            self._signal_program['u_x_max'] = x_max_new
            
            self._mads_program['u_x_min'] = x_min_new
            self._mads_program['u_x_max'] = x_max_new

            self._box_program['u_x_min'] = x_min_new
            self._box_program['u_x_max'] = x_max_new

            y_min_new = self._signal_program['u_y_min'] * np.exp(dx)
            y_max_new = self._signal_program['u_y_max'] * np.exp(dx)
            
            self._signal_program['u_y_min'] = y_min_new
            self._signal_program['u_y_max'] = y_max_new
            
            self._mads_program['u_y_min'] = y_min_new
            self._mads_program['u_y_max'] = y_max_new

            self._box_program['u_y_min'] = y_min_new
            self._box_program['u_y_max'] = y_max_new
    
        # # TODO emit signal to update the spin box.
    
        self.update()
    
        return


    def on_mouse_move(self, event):
    
        if event.press_event is None:
            return

        modifiers = event.modifiers
        p1 = event.press_event.pos
        p2 = event.pos

        p1 = np.array(event.last_event.pos)[:2]
        p2 = np.array(event.pos)[:2]
        
        dx, dy = 0.1*(p1 - p2)

        self._box_program['u_x_min'] += dx
        self._box_program['u_x_max'] += dx
        self._box_program['u_y_min'] += dy
        self._box_program['u_y_max'] += dy

        self._signal_program['u_x_min'] += dx
        self._signal_program['u_x_max'] += dx
        self._signal_program['u_y_min'] += dy
        self._signal_program['u_y_max'] += dy

        self._mads_program['u_x_min'] += dx
        self._mads_program['u_x_max'] += dx
        self._mads_program['u_y_min'] += dy
        self._mads_program['u_y_max'] += dy


        # # TODO emit signal to update the spin box.
    
        self.update()
        return

    def on_draw(self, event):

        _ = event
        gloo.clear()
        self._signal_program.draw('line_strip')
        self._mads_program.draw('line_strip')
        # self._peaks_program.draw('line_strip')
        self._box_program.draw('line_strip')

        return

    def on_reception(self, data, mads, peaks):

        # TODO find a better solution for the 2 following lines.
        if data.shape[1] > self.nb_signals:
            data = data[:, :self.nb_signals]

        k = self._nb_samples_per_buffer

        self._signal_values[:, :-k] = self._signal_values[:, k:]
        self._signal_values[:, -k:] = np.transpose(data)
        signal_values = self._signal_values.ravel().astype(np.float32)

        self._signal_program['a_signal_value'].set_data(signal_values)

        self._mads_values[:, :-2] = self._mads_values[:, 2:]
        if mads is not None:
            self._mads_values[:, -2:] = np.transpose(np.tile(mads, reps=(2, 1)))
        else:
            self._mads_values[:, -2:] = self._mads_values[:, -4:-2]
        mads_values = self._mads_values.ravel().astype(np.float32)

        self._mads_program['a_mads_value'].set_data(self.mad_factor * mads_values)

        if peaks is not None:
            peaks_channels = np.concatenate([i*np.ones(len(peaks[i]), dtype=np.float32) for i in peaks.keys()])
            peaks_values = np.concatenate([peaks[i].astype(np.float32) for i in peaks.keys()]) 
            peaks_positions = np.ascontiguousarray(np.vstack((peaks_values, peaks_channels)).T)
            peaks_sizes = 10*self.pixel_scale*np.ones(len(peaks_positions), dtype=np.float32)
            self._peaks_program['a_peaks_position'].set_data(peaks_positions)
            self._peaks_program['a_peaks_sizes'].set_data(peaks_sizes)
            #self._peaks_program['a_peaks_color'] = gloo.VertexBuffer(peaks_colors)

        mads_thresholds = np.repeat(np.mean(np.reshape(mads_values, (self.nb_signals, -1))
                                            , axis=1), repeats=20480)
        self._signal_program['a_spike_threshold'] = mads_thresholds * self.mad_factor
        self.update()

        return

    def set_time(self, value):

        t_scale = self._time_max / value
        self._signal_program['u_t_scale'] = t_scale
        self._mads_program['u_t_scale'] = t_scale
        self.update()

        return

    def set_voltage(self, value):

        v_scale = value
        self._signal_program['u_v_scale'] = v_scale
        self._mads_program['u_v_scale'] = v_scale
        self.update()

        return

    def set_mads(self, value):

        self.mad_factor = value
        self.update()

        return

    def set_channels(self, channels):

        self.channels = channels
        self.update()

        return

    def show_mads(self, value):

        self._mads_program['display'] = value
        self.update()

        return

    def show_peaks(self, value):

        self._peaks_program['display'] = value
        self.update()

        return

    def dsp_spikes_color(self, s):
        if s == 2:
            self._signal_program['see_spikes'] = 1.0
        else:
            self._signal_program['see_spikes'] = 0.0
        self.update()