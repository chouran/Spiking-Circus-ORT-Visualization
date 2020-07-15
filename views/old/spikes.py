import numpy as np

from vispy import app, gloo, scene, visuals
from vispy.util import keys

from circusort.io.probe import load_probe

from views.canvas import ViewCanvas
from views.programs import LinesPlot, ScatterPlot
from utils.widgets import Controler


class RasterSpikes(visuals.Visual):
    requires = ['spike_times', 'nb_template']
    name = "Spike Times"

    vertex = """
    attribute vec3 a_spike_times;
    // a_spike_times.x = spike_times;
    // a_spike_times.y = nb_template;
    // a_spike_times.z = nb_electrode;
    attribute vec3 a_template_color;
    uniform float u_nb_template;
    uniform float u_time;
    uniform float radius,
    
    varying vec4 v_fg_color;
    varying vec4 v_bg_color;
    varying float v_linewidth;
    varying float v_antialias;
    varying float v_radius;

    void main() {
        //position
        x = -1.0 + (2.0 * spike_times.x / u_time);
        y = -0.9 + (1.8 * (spike_times.y) / (u_nb_template+1)); 
        
        gl_Position = vec4(x, y, 0.0, 1.0);
        
        v_linewidth = 1.0;
        v_antialias = 1.0;
        v_fg_color  = vec4(1.0, 1.0, 1.0, 0.5);
        v_bg_color  = vec4(a_spike_color, 1.0);
        v_radius = radius;
        gl_PointSize = 2.0*(radius + v_linewidth + 1.5*v_antialias);
    }
    """

    fragment = """
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

    def __init__(self, probe_path=None, params=None):
        self._time = 1
        self._nb_templates = 1
        self._nb_templates_selected = 0
        self._color = [1.0, 1.0, 1.0]
        self._spike_times = [0.5, 1, 1]
        self._initialize = False

        visuals.Visual.__init__(self, vcode=self.vertex, fcode=self.fragment)

        self.freeze()


        self.shared_program['u_time'] = self._time
        self.shared_program['u_nb_template'] = self._nb_templates
        self.shared_program['a_spike_times'] = self._time
        self.shared_program['a_template_color'] = self._color

        self._draw_mode = 'points'

    def _on_reception(self, data):
        self._time = data['time'] if 'time' in data else None
        self._nb_template = data['nb_template'] if 'nb_template' in data else None
        spike_times = data['spike_times'] if 'spike_times' in data else None

        if not self._initialize:
            colors = self.get_colors(self._nb_templates)
            nb_init_spike = len(spike_times)
            self._spike_times = np.array(spike_times)
            self._color = np.zeros((nb_init_spike, 3))

            for l in spike_times:
                i = 0
                self._color[i] = colors[l[1] - 1]
            self._initialize = True

        else:
            colors = self.get_colors(self._nb_templates)

            for l in spike_times:
                self._spike_times = np.concatenate(self._spike_times, l)
                self._color = np.concatenate(self._color, colors[l[1] - 1])

        self.shared_program['a_spike_times'] = self._spike_times
        self.shared_program['a_template_color'] = self._color
