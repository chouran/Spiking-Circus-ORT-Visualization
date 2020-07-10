import numpy as np

from vispy import app, gloo
from vispy.util import keys

from circusort.io.probe import load_probe

from views.canvas import ViewCanvas
from views.programs import LinesPlot, ScatterPlot
from utils.widgets import Controler

SPIKES_VERT_SHADER = """
attribute vec3 a_spike_time;
// a_spike_time.x = spike_times;
// a_spike_time.y = nb_template;
// a_spike_time.z = nb_electrode;
attribute vec3 a_template_color;
uniform float u_nb_template;
uniform float u_time;
uniform float u_radius;

varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_linewidth;
varying float v_antialias;
varying float v_radius;

void main() {
    //position
    float x = -1.0 + (2.0 * a_spike_time.x / u_time) - 0.1;
    float y = -0.9 + (1.8 * (a_spike_time.y+1) / (u_nb_template+1)); 

    gl_Position = vec4(x, y, 0.0, 1.0);

    v_linewidth = 1.0;
    v_antialias = 1.0;
    v_fg_color  = vec4(1.0, 1.0, 1.0, 0.5);
    v_bg_color  = vec4(a_template_color, 1.0);
    v_radius = u_radius;
    gl_PointSize = 2.0*(u_radius + v_linewidth + 1.5*v_antialias);
}
"""

SPIKES_FRAG_SHADER = """
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
        self._colors = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        self._spike_times = np.array([[0.5, 1.0, 1.0]], dtype=np.float32)
        self._initialize = False
        self._radius = 5

        self.programs['spikes'] = ScatterPlot(SPIKES_VERT_SHADER, SPIKES_FRAG_SHADER)
        self.programs['spikes']['a_spike_time'] = self._spike_times
        self.programs['spikes']['a_template_color'] = self._colors
        self.programs['spikes']['u_time'] = self._time
        self.programs['spikes']['u_nb_template'] = self._nb_templates
        self.programs['spikes']['u_radius'] = self._radius

    def _on_reception(self, data):
        self._time = data['time']
        self._nb_templates = data['nb_templates']
        spike_times = data['spike_times'] if 'spike_times' in data else None
        self._color_list = self.get_colors(self._nb_templates)

        if not self._initialize and spike_times is not None:
            nb_init_spike = len(spike_times)
            self._spike_times = np.array(spike_times).astype(np.float32)
            self._colors = np.zeros((nb_init_spike, 3)).astype(np.float32)

            for l in spike_times:
                i = 0
                self._colors[i] = self._color_list[l[1]]
                i += 1
            self._initialize = True

        elif spike_times is not None:
            for l in spike_times:
                new_spike = np.reshape(np.array(l, dtype=np.float32), (-1, 3))
                new_color = np.reshape(self._color_list[l[1]], (-1, 3))

                self._spike_times = np.concatenate((self._spike_times, new_spike))
                self._colors = np.concatenate((self._colors, new_color))

                print('st', self._spike_times)
                print('col', self._colors)

        self.programs['spikes']['a_spike_time'] = self._spike_times
        self.programs['spikes']['a_template_color'] = self._colors
        self.programs['spikes']['u_time'] = self._time
        self.programs['spikes']['u_nb_template'] = self._nb_templates

        self.update()

        return
