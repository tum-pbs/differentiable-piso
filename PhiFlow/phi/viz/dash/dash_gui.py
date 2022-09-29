# coding=utf-8
import dash_core_components as dcc
import dash_html_components as html
import six

from phi.struct.tensorop import collapsed_gather_nd
from .board import build_benchmark, build_tf_profiler, build_tensorboard_launcher, build_system_controls
from .log import build_log
from .model_controls import build_model_controls
from .viewsettings import build_view_selection
from .dash_app import DashApp
from .info import build_app_details, build_description, build_phiflow_info, build_app_time
from .viewer import build_viewer
from .player_controls import build_status_bar, build_player_controls
from ..display import AppDisplay


class DashGui(AppDisplay):

    def __init__(self, app):
        AppDisplay.__init__(self, app)
        self.dash_app = None

    def setup(self):
        header_layout = html.Div([
                dcc.Link('Home', href='/'),
                ' - ',
                dcc.Link('Side-by-Side', href='/side-by-side'),
                ' - ',
                dcc.Link('Quad', href='/quad'),
                ' - ',
                dcc.Link('Info', href='/info'),
                ' - ',
                dcc.Link('Log', href='/log'),
                ' - ',
                dcc.Link(u'Φ Board', href='/board'),
                # ' - ',
                # dcc.Link('Scripting', href='/scripting'),
                ' - ',
                html.A('Help', href='https://github.com/tum-pbs/PhiFlow/blob/master/documentation/Web_Interface.md', target='_blank'),
            ])
        dash_app = self.dash_app = DashApp(self.app, self.config, header_layout)

        # --- Shared components ---
        player_controls = build_player_controls(dash_app)
        status_bar = build_status_bar(dash_app)
        model_controls = build_model_controls(dash_app)

        # --- Home ---
        layout = html.Div([
            build_description(dash_app),
            build_view_selection(dash_app),
            html.Div(style={'width': 1000, 'height': 800, 'margin-left': 'auto', 'margin-right': 'auto'}, children=[
                build_viewer(dash_app, id='home', initial_field_name=collapsed_gather_nd(self.config.get('display', None), [0]), config=self.config),
            ]),
            status_bar,
            player_controls,
            model_controls,
        ])
        dash_app.add_page('/', layout)

        # --- Side by Side ---
        if 'display' in self.config:
            sbs_fieldnames = self.config['display']
            if sbs_fieldnames is None:
                sbs_fieldnames = [None, None]
            elif isinstance(sbs_fieldnames, six.string_types):
                sbs_fieldnames = [sbs_fieldnames] * 2
        else:
            sbs_fieldnames = self.app.fieldnames
            if len(sbs_fieldnames) == 0:
                sbs_fieldnames = ['None', 'None']
            elif len(sbs_fieldnames) < 2:
                sbs_fieldnames = list(sbs_fieldnames) + ['None']
        layout = html.Div([
            build_view_selection(dash_app),
            html.Div(style={'width': '50%', 'height': 700, 'display': 'inline-block'}, children=[
                build_viewer(dash_app, id='left', initial_field_name=sbs_fieldnames[0], config=self.config),
            ]),
            html.Div(style={'width': '50%', 'height': 700, 'display': 'inline-block'}, children=[
                build_viewer(dash_app, id='right', initial_field_name=sbs_fieldnames[1], config=self.config),
            ]),
            status_bar,
            player_controls,
            model_controls,
        ])
        dash_app.add_page('/side-by-side', layout)

        # --- Quad ---
        layout = html.Div([
            build_view_selection(dash_app),
            html.Div(style={'width': '50%', 'height': 700, 'display': 'inline-block'}, children=[
                build_viewer(dash_app, id='top-left', initial_field_name=sbs_fieldnames[0], config=self.config),
            ]),
            html.Div(style={'width': '50%', 'height': 700, 'display': 'inline-block'}, children=[
                build_viewer(dash_app, id='top-right', initial_field_name=sbs_fieldnames[1], config=self.config),
            ]),
            html.Div(style={'width': '50%', 'height': 700, 'display': 'inline-block'}, children=[
                build_viewer(dash_app, id='bottom-left', initial_field_name=sbs_fieldnames[0], config=self.config),
            ]),
            html.Div(style={'width': '50%', 'height': 700, 'display': 'inline-block'}, children=[
                build_viewer(dash_app, id='bottom-right', initial_field_name=sbs_fieldnames[1], config=self.config),
            ]),
            status_bar,
            player_controls,
            model_controls,
        ])
        dash_app.add_page('/quad', layout)

        # --- Log ---
        layout = html.Div([
            dcc.Markdown('# Log'),
            status_bar,
            player_controls,
            build_log(dash_app)
        ])
        dash_app.add_page('/log', layout)

        # --- Info ---
        layout = html.Div([
            build_description(dash_app),
            status_bar,
            player_controls,
            build_phiflow_info(dash_app),
            build_app_details(dash_app),
            build_app_time(dash_app),
        ])
        dash_app.add_page('/info', layout)

        # --- Board ---
        layout = html.Div([
            dcc.Markdown(u'# Φ Board'),
            status_bar,
            player_controls,
        ] + ([] if 'tensorflow' not in dash_app.app.traits else [
            build_tensorboard_launcher(dash_app),
        ]) + [
            model_controls,
            build_benchmark(dash_app),
        ] + ([] if 'tensorflow' not in dash_app.app.traits else [
            build_tf_profiler(dash_app),
        ]) + [
            build_system_controls(dash_app),
            # ToDo: Graphs, Record/Animate
        ])
        dash_app.add_page('/board', layout)

        # --- Scripting ---
        layout = html.Div([
            dcc.Markdown(u'# Python Scripting'),
            'Custom Fields, Execute script, Restart'
        ])
        dash_app.add_page('/scripting', layout)

        return self.dash_app.dash

    def show(self, caller_is_main):
        if not caller_is_main and self.config.get('external_web_server', False):
            return self.dash_app.server
        else:
            port = self.config.get('port', 8051)
            print('Starting Dash server on http://localhost:%d/' % port)
            self.dash_app.dash.run_server(debug=True, host='0.0.0.0', port=port, use_reloader=False)
            return self
