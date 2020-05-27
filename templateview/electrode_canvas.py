import numpy as np

from vispy import app, gloo
from vispy.util import keys

from circusort.io.probe import load_probe
from circusort.io.template import load_template_from_dict

BOUNDARY_VERT_SHADER = """
// Coordinates of the position of the box.
attribute vec2 a_pos_probe;
// Coordinates of the position of the corner.
attribute vec2 a_corner_position;
// Uniform variables used to transform the subplots.
uniform float u_x_min;
uniform float u_x_max;
uniform float u_y_min;
uniform float u_y_max;
uniform float u_d_scale;


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
    vec2 b = vec2(-1.0 + 2.0 * (a_pos_probe.x - u_x_min) / w,
                    -1.0 + 2.0 * (a_pos_probe.y - u_y_min) / h);
    // Apply the transformation.
    gl_Position = vec4(a * p + b, 0.0, 1.0);
}
"""

CHANNELS_VERT_SHADER = """
//uniform vec2 resolution;
//attribute vec2 channel_centers;
attribute vec2 a_channel_position;
attribute float a_selected_channel;

uniform float u_x_min;
uniform float u_x_max;
uniform float u_y_min;
uniform float u_y_max;
uniform float u_d_scale;
uniform float radius;

varying vec2 v_center;
varying float v_radius; 
varying float v_select;

varying vec4 v_fg_color;
varying vec4 v_selec_color;
varying vec4 v_unsel_color;
varying float v_linewidth;
varying float v_antialias;

void main(){

    float w = u_x_max - u_x_min;
    float h = u_y_max - u_y_min;
    //vec2 p = (0.0, 0.0);
    vec2 a = vec2(1.0 / (1.0 + w / u_d_scale), 1.0 / (1.0 + h / u_d_scale));
    vec2 b = vec2(-1.0 + 2.0 * (a_channel_position.x - u_x_min) / w, -1.0 + 2.0 * (a_channel_position.y - u_y_min) / h);
    //center = vec2(a * p + b);
    vec2 center = b;
    v_center = center;
    v_select = a_selected_channel;
    v_radius = radius;
    
    v_linewidth = 1.0;
    v_antialias = 1.0;
    v_fg_color  = vec4(1.0,1.0,1.0,0.5);
    v_selec_color = vec4(0.7, 0.7, 0.7, 1.0);
    v_unsel_color = vec4(0.3, 0.3, 0.3, 1.0);
    
    gl_PointSize = 2.0*(v_radius + v_linewidth + 1.5*v_antialias);
    gl_Position = vec4(center, 0.0, 1.0);
}
"""

BARYCENTER_VERT_SHADER = """
attribute vec2 a_barycenter_position;
attribute float a_selected_template;
attribute vec3 a_color;

uniform float u_x_min;
uniform float u_x_max;
uniform float u_y_min;
uniform float u_y_max;
uniform float u_d_scale;
uniform float radius;

varying float v_radius; 
varying float v_selected_temp;

varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_linewidth;
varying float v_antialias;

void main() {
    float w = u_x_max - u_x_min;
    float h = u_y_max - u_y_min;
    vec2 a_bis = vec2(w , h);
    vec2 p = a_barycenter_position;
    vec2 a = vec2(1.0 / (1.0 + w / u_d_scale), 1.0 / (1.0 + h / u_d_scale));
    //gl_PointSize = 2.0 + ceil(2.0*radius);
    //gl_PointSize  = radius;
    //TODO modify the following with parameters
    gl_Position = vec4(p/135, 0.0, 1.0);
    
    v_linewidth = 1.0;
    v_antialias = 1.0;
    v_fg_color  = vec4(1.0, 1.0, 1.0, 0.5);
    v_bg_color  = vec4(a_color,    1.0);
    gl_PointSize = 2.0*(radius + v_linewidth + 1.5*v_antialias);    
      
    v_selected_temp = a_selected_template;
    v_radius = radius;
}
"""

BOUNDARY_FRAG_SHADER = """
// Fragment shader.
void main() {
    gl_FragColor = vec4(0.25, 0.25, 0.25, 1.0);
}
"""

CHANNELS_FRAG_SHADER = """
varying vec2 v_center;
varying float v_radius;
varying float v_select;

varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_linewidth;
varying float v_antialias;
varying vec4 v_selec_color;
varying vec4 v_unsel_color;
// Fragment shader.
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
        {
            //gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
            if (v_select == 1)
                gl_FragColor = vec4(0.1, 1.0, 0.1, alpha);
            else
                gl_FragColor = vec4(0.3, 0.3, 0.3, alpha);        
        }
    }
}
"""

BARYCENTER_FRAG_SHADER = """
varying float v_radius;
varying float v_selected_temp;

varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_linewidth;
varying float v_antialias;

// Fragment shader.
void main() {
    float size = 2.0*(v_radius + v_linewidth + 1.5*v_antialias);
    float t = v_linewidth/2.0-v_antialias;
    float r = length((gl_PointCoord.xy - vec2(0.5,0.5))*size);
    float d = abs(r - v_radius) - t;
    
    if (v_selected_temp == 0.0)
        discard;
    else 
    {
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
}
"""

class MEACanvas(app.Canvas):

    def __init__(self, probe_path=None, params=None):
        app.Canvas.__init__(self, title="Probe view")

        self.probe = load_probe(probe_path)
        # self.channels = params['channels']
        self.nb_channels = self.probe.nb_channels
        self.initialized = False

        # TODO Add method to probe file to extract minimum coordinates without the interelectrode dist
        x_min, x_max = self.probe.x_limits[0] + self.probe.minimum_interelectrode_distance,\
                       self.probe.x_limits[1] - self.probe.minimum_interelectrode_distance
        y_min, y_max = self.probe.y_limits[0] + self.probe.minimum_interelectrode_distance, \
                       self.probe.y_limits[1] - self.probe.minimum_interelectrode_distance

        probe_corner = np.array([[x_max, y_max],
                                 [x_min, y_max],
                                 [x_min, y_min],
                                 [x_max, y_min],
                                 [x_max, y_max]], dtype=np.float32)

        corner_bound_positions = np.array([[+1.0, +1.0],
                                           [-1.0, +1.0],
                                           [-1.0, -1.0],
                                           [+1.0, -1.0],
                                           [+1.0, +1.0]], dtype=np.float32)

        # Define GLSL program.
        self._boundary_program = gloo.Program(vert=BOUNDARY_VERT_SHADER, frag=BOUNDARY_FRAG_SHADER)
        self._boundary_program['a_pos_probe'] = probe_corner
        self._boundary_program['a_corner_position'] = corner_bound_positions
        self._boundary_program['u_x_min'] = self.probe.x_limits[0]
        self._boundary_program['u_x_max'] = self.probe.x_limits[1]
        self._boundary_program['u_y_min'] = self.probe.y_limits[0]
        self._boundary_program['u_y_max'] = self.probe.y_limits[1]
        self._boundary_program['u_d_scale'] = self.probe.minimum_interelectrode_distance

        # Probe
        channel_pos = np.c_[
            np.repeat(self.probe.x.astype(np.float32), repeats=1),
            np.repeat(self.probe.y.astype(np.float32), repeats=1),
        ]
        selected_channels = np.ones(self.nb_channels, dtype=np.float32)

        self._channel_program = gloo.Program(vert=CHANNELS_VERT_SHADER, frag=CHANNELS_FRAG_SHADER)
        self._channel_program['a_channel_position'] = channel_pos
        self._channel_program['a_selected_channel'] = selected_channels
        self._channel_program['radius'] = 10
        self._channel_program['u_x_min'] = self.probe.x_limits[0]
        self._channel_program['u_x_max'] = self.probe.x_limits[1]
        self._channel_program['u_y_min'] = self.probe.y_limits[0]
        self._channel_program['u_y_max'] = self.probe.y_limits[1]
        self._channel_program['u_d_scale'] = self.probe.minimum_interelectrode_distance
        #self._channel_program['u_d_scale'] = self.probe.minimum_interelectrode_distance

        #Barycenters
        self.nb_templates = 0
        barycenter_position = np.zeros((self.nb_templates, 2), dtype=np.float32)
        temp_selected = np.ones(self.nb_templates, dtype=np.float32)
        self.barycenter = np.zeros((self.nb_templates, 2), dtype=np.float32)
        
        np.random.seed(12)
        self.bary_color = np.random.uniform(size=(self.nb_templates, 3), low=.5, high=.9).astype(np.float32)

        self._barycenter_program = gloo.Program(vert=BARYCENTER_VERT_SHADER, frag=BARYCENTER_FRAG_SHADER)
        self._barycenter_program['a_barycenter_position'] = self.barycenter
        self._barycenter_program['a_selected_template'] = temp_selected
        self._barycenter_program['a_color'] = self.bary_color
        self._barycenter_program['radius'] = 5
        self._barycenter_program['u_x_min'] = self.probe.x_limits[0]
        self._barycenter_program['u_x_max'] = self.probe.x_limits[1]
        self._barycenter_program['u_y_min'] = self.probe.y_limits[0]
        self._barycenter_program['u_y_max'] = self.probe.y_limits[1]
        self._barycenter_program['u_d_scale'] = self.probe.minimum_interelectrode_distance

        # Final details.
        gloo.set_viewport(0, 0, *self.physical_size)
        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    @staticmethod
    def on_resize(event):
        gloo.set_viewport(0, 0, *event.physical_size)
        print("mea resize")
        return

    def on_draw(self, event):
        __ = event
        gloo.clear()
        self._boundary_program.draw('line_strip')
        self._channel_program.draw('points')
        self._barycenter_program.draw('points')
        return

    def selected_channels(self, L):
        channels_selected = np.zeros(self.nb_channels, dtype=np.float32)
        # Remove redundant channels
        for i in set(L):
            channels_selected[i] = 1
        self._channel_program['a_selected_channel'] = channels_selected
        self.update()
        return

    def selected_templates(self, L):
        template_selected = np.zeros(self.nb_templates, dtype=np.float32)
        for i in (L):
            template_selected[i] = 1
        self._barycenter_program['a_selected_template'] = template_selected
        self.update()
        return

    def on_reception_bary(self, bar, nb_template):
        self.nb_templates = nb_template
        
        if bar is not None:
            for b in bar:
                self.barycenter = np.vstack((self.barycenter, np.array(b, dtype=np.float32)))

            temp_selected = np.ones(self.nb_templates, dtype=np.float32)
            np.random.seed(12)
            self.bary_color = np.random.uniform(size=(self.nb_templates, 3), low=0.3, high=.9).astype(np.float32)

            self._barycenter_program['a_barycenter_position'] = self.barycenter
            self._barycenter_program['a_selected_template'] = temp_selected
            self._barycenter_program['a_color'] = self.bary_color
            self.update()
        return




