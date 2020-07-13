import numpy as np

from vispy import app, gloo
from vispy.util import keys

from circusort.io.probe import load_probe

from views.canvas import ViewCanvas
from views.programs import LinesPlot, ScatterPlot
from utils.widgets import Controler

SPIKES_VERT_SHADER = """
attribute vec3 a_spike_time;
attribute vec2 a_template_index;
// a_spike_time.x = spike_times;
// a_spike_time.y = nb_template;
// a_spike_time.z = nb_electrode;
attribute vec3 a_template_color;
uniform float u_nb_template_selected;
uniform float u_time;
uniform float u_radius;

varying vec2 v_template_index;
varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_linewidth;
varying float v_antialias;
varying float v_radius;

void main() {
    //position
    float x = -1.0 + (2.0 * a_spike_time.x / u_time) - 0.1;
    float y = -0.9 + (1.8 * (a_template_index.y) / (u_nb_template_selected+1)); 

    gl_Position = vec4(x, y, 0.0, 1.0);

    v_linewidth = 1.0;
    v_antialias = 1.0;
    v_fg_color  = vec4(1.0, 1.0, 1.0, 0.5);
    v_bg_color  = vec4(a_template_color, 1.0);
    v_radius = u_radius;
    v_template_index = a_template_index;
    gl_PointSize = 2.0*(u_radius + v_linewidth + 1.5*v_antialias);
}
"""

SPIKES_FRAG_SHADER = """
varying vec2 v_template_index;
varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_linewidth;
varying float v_antialias;
varying float v_radius;

void main() {
    float size = 2.0*(v_radius + v_linewidth + 1.5*v_antialias);
    float t = v_linewidth/2.0-v_antialias;
    float r = length((gl_PointCoord.xy - vec2(0.5,0.5))*size);
    float d = abs(r - v_radius) - t;

    if (v_template_index.y < 0)
        discard;
    if( d < 0.0 )
        gl_FragColor = v_fg_color;
    else
    {
        float alpha = d/v_antialias;
        alpha = exp(-alpha*alpha);
        if (r > v_radius)
            gl_FragColor = vec4(v_fg_color.rgb, alpha*v_fg_color.a);
        else
            gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
    }
}
"""


class RasterSpikesGloo(ViewCanvas):
    requires = ['spike_times', 'nb_templates', 'time']
    name = "Spike Times Gloo"

    def __init__(self, probe_path=None, params=None):

        ViewCanvas.__init__(self, probe_path, title="Spikes View", box='single')

        self._time = 1
        self._nb_templates = 1
        self._nb_templates_selected = 0
        self._dic_selected_cell = {}
        self._colors = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        self._spike_times = np.array([[0.5, 1.0, 1.0]], dtype=np.float32)
        self._template_index = np.array([[0, 1]], dtype=np.float32)
        self._initialize = False
        self._radius = 5

        self.programs['spikes'] = ScatterPlot(SPIKES_VERT_SHADER, SPIKES_FRAG_SHADER)
        self.programs['spikes']['a_spike_time'] = self._spike_times
        self.programs['spikes']['a_template_color'] = self._colors
        self.programs['spikes']['u_time'] = self._time
        self.programs['spikes']['u_nb_template_selected'] = self._nb_templates_selected
        self.programs['spikes']['u_radius'] = self._radius

    def _highlight_selection(self, selection):
        self._nb_templates_selected = len(selection)
        self._dic_selected_cell = {}
        self._template_index[:, 1] = -1
        temp_position = 1
        for nb_template in selection:
            self._dic_selected_cell[nb_template] = temp_position
            self._template_index[np.where(self._template_index[:, 0] == nb_template), 1] = temp_position
            temp_position += 1
        print(self._dic_selected_cell)

        self.programs['spikes']['u_nb_template_selected'] = self._nb_templates_selected
        self.programs['spikes']['a_template_index'] = self._template_index

        return

    def _on_reception(self, data):
        self._time = data['time']
        self._nb_templates = data['nb_templates']
        spike_times = data['spike_times'] if 'spike_times' in data else None
        self._color_list = self.get_colors(self._nb_templates)

        if not self._initialize and spike_times is not None:
            nb_init_spike = len(spike_times)
            self._nb_templates_selected = self._nb_templates
            self._spike_times = np.array(spike_times).astype(np.float32)
            self._colors = np.zeros((nb_init_spike, 3)).astype(np.float32)
            list_index_init = []

            i, j = 0, 1
            for st in spike_times:
                self._colors[i] = self._color_list[st[1]]
                list_index_init.append(st[1])
                i += 1
            for k in set(list_index_init):
                self._dic_selected_cell[k] = j
                j += 1
            self._initialize = True

        elif spike_times is not None:
            for st in spike_times:
                new_spike = np.reshape(np.array(st, dtype=np.float32), (-1, 3))
                new_color = np.reshape(self._color_list[st[1]], (-1, 3))
                nb_template = st[1]
                if nb_template in self._dic_selected_cell.keys():
                    index_plot = self._dic_selected_cell[nb_template]
                    self._template_index = np.concatenate((self._template_index,
                                                           np.array([[nb_template, index_plot]],
                                                                    dtype=np.float32)))
                else:
                    self._template_index = np.concatenate((self._template_index,
                                                           np.array([[nb_template, -1]],
                                                                    dtype=np.float32)))
                self._spike_times = np.concatenate((self._spike_times, new_spike))
                self._colors = np.concatenate((self._colors, new_color))

        self.programs['spikes']['a_spike_time'] = self._spike_times
        self.programs['spikes']['a_template_index'] = self._template_index
        self.programs['spikes']['a_template_color'] = self._colors
        self.programs['spikes']['u_time'] = self._time
        self.programs['spikes']['u_nb_template_selected'] = self._nb_templates_selected

        self.update()

        return


class RasterSpikes(Controler):

    def __init__(self, canvas, bin_size=0.1):
        '''
        Control widgets:
        '''

        Controler.__init__(self, canvas)
        self.bin_size = bin_size

        self.dsb_bin_size = self.double_spin_box(label='Bin Size', unit='seconds', min_value=0.01,
                                                 max_value=100, step=0.1, init_value=self.bin_size)

        self.dsb_zoom = self.double_spin_box(label='Zoom', min_value=1, max_value=50, step=0.1,
                                             init_value=1)
        self.dsb_time_window = self.double_spin_box(label='Time window', unit='seconds',
                                                    min_value=1, max_value=50, step=0.1,
                                                    init_value=1)
        self.cb_tw = self.checkbox(label='Time window from start', init_state=True)

        self.add_widget(self.dsb_bin_size, self._on_binsize_changed)
        self.add_widget(self.dsb_zoom, self._on_zoom_changed)
        self.add_widget(self.dsb_time_window, self._on_time_window_changed)
        self.add_widget(self.cb_tw, self._time_window_rate_full)

    def _on_binsize_changed(self, bin_size):
        self.bin_size = self.dsb_bin_size['widget'].value()
        return

    def _on_zoom_changed(self):
        zoom_value = self.dsb_zoom['widget'].value()
        self.canvas.zoom(zoom_value)
        return

    def _time_window_rate_full(self):
        value = self.cb_tw['widget'].isChecked()
        self.canvas.set_value({"full": value})
        return

    def _on_time_window_changed(self):
        tw_value = self.dsb_time_window['widget'].value()
        self.canvas.set_value({"range": (tw_value, self.bin_size)})
        return
