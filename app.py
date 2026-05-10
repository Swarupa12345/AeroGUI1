import PySimpleGUI as sg

from predictor import aerodynamic_prediction
from optimizer import run_optimization
from envelope import alpha_sweep, mach_sweep


sg.theme('skyblue')


# =========================================================
# INPUT PARAMETERS
# =========================================================

input_layout = [

    [sg.Text('Nose Length'), sg.Input(key='nose_len')],
    [sg.Text('Body Length'), sg.Input(key='body_len')],
    [sg.Text('Wing LE'), sg.Input(key='wing_le')],
    [sg.Text('Root Chord'), sg.Input(key='root_chord')],
    [sg.Text('Tip Chord'), sg.Input(key='tip_chord')],
    [sg.Text('Semi Span'), sg.Input(key='semi_span')],
    [sg.Text('Root Thickness'), sg.Input(key='root_th')],
    [sg.Text('Tip Thickness'), sg.Input(key='tip_th')],
    [sg.Text('Wing Sweep'), sg.Input(key='wing_sweep')],

    [sg.Text('Tail LE'), sg.Input(key='tail_le')],
    [sg.Text('Root Chord 1'), sg.Input(key='root_chord1')],
    [sg.Text('Tip Chord 1'), sg.Input(key='tip_chord1')],
    [sg.Text('Semi Span 1'), sg.Input(key='semi_span1')],
    [sg.Text('Root Thickness 1'), sg.Input(key='root_th1')],
    [sg.Text('Tip Thickness 1'), sg.Input(key='tip_th1')],

    [sg.Text('Mach'), sg.Input(key='mach')],
    [sg.Text('Alpha'), sg.Input(key='alpha')],
    [sg.Text('Altitude'), sg.Input(key='alt')]
]


# =========================================================
# TAB 1 : SIMPLE PREDICTION
# =========================================================
prediction_tab = [

    [

        sg.Column(

            input_layout,

            scrollable=True,

            vertical_scroll_only=True,

            size=(400, 600)

        ),

        sg.VSeparator(),

        sg.Column(

            [

                [sg.Text(
                    'Predicted Outputs',
                    font=('Any', 16)
                )],

                [sg.HorizontalSeparator()],


                [sg.Text('CL :',
                         size=(15,1)),

                 sg.Text(
                     '',
                     key='cl_out',
                     size=(30,1),
                     font=('Any', 14)
                 )],


                [sg.Text('CD :',
                         size=(15,1)),

                 sg.Text(
                     '',
                     key='cd_out',
                     size=(30,1),
                     font=('Any', 14)
                 )],


                [sg.Text('XCP :',
                         size=(15,1)),

                 sg.Text(
                     '',
                     key='xcp_out',
                     size=(30,1),
                     font=('Any', 14)
                 )],


                [sg.HorizontalSeparator()],


                [

                    sg.Button(
                        'Estimate',
                        size=(15,2)
                    ),

                    sg.Button(
                        'Clear',
                        size=(15,2)
                    ),

                    sg.Button(
                        'Exit',
                        size=(15,2)
                    )

                ]

            ],

            size=(700, 600)

        )

    ]

]

# =========================================================
# TAB 2 : OPTIMIZATION
# =========================================================

optimization_tab = [

    [sg.Text('Optimization Design Variables',
             font=('Any', 14))],

    [
        sg.Text('Parameter', size=(20,1)),
        sg.Text('Lower Bound', size=(12,1)),
        sg.Text('Upper Bound', size=(12,1))
    ],

    [sg.Text('Nose Length', size=(20,1)),
     sg.Input('1', size=(12,1), key='nose_lower'),
     sg.Input('10', size=(12,1), key='nose_upper')],

    [sg.Text('Body Length', size=(20,1)),
     sg.Input('5', size=(12,1), key='body_lower'),
     sg.Input('30', size=(12,1), key='body_upper')],

    [sg.Text('Wing LE', size=(20,1)),
     sg.Input('1', size=(12,1), key='wingle_lower'),
     sg.Input('10', size=(12,1), key='wingle_upper')],

    [sg.Text('Root Chord', size=(20,1)),
     sg.Input('1', size=(12,1), key='root_lower'),
     sg.Input('10', size=(12,1), key='root_upper')],

    [sg.Text('Tip Chord', size=(20,1)),
     sg.Input('1', size=(12,1), key='tip_lower'),
     sg.Input('10', size=(12,1), key='tip_upper')],

    [sg.Text('Semi Span', size=(20,1)),
     sg.Input('1', size=(12,1), key='span_lower'),
     sg.Input('20', size=(12,1), key='span_upper')],

    [sg.Text('Root Thickness', size=(20,1)),
     sg.Input('0.1', size=(12,1), key='rootth_lower'),
     sg.Input('1', size=(12,1), key='rootth_upper')],

    [sg.Text('Tip Thickness', size=(20,1)),
     sg.Input('0.1', size=(12,1), key='tipth_lower'),
     sg.Input('1', size=(12,1), key='tipth_upper')],

    [sg.Text('Wing Sweep', size=(20,1)),
     sg.Input('10', size=(12,1), key='sweep_lower'),
     sg.Input('60', size=(12,1), key='sweep_upper')],

    [sg.Text('Tail LE', size=(20,1)),
     sg.Input('1', size=(12,1), key='tail_lower'),
     sg.Input('10', size=(12,1), key='tail_upper')],

    [sg.Text('Root Chord 1', size=(20,1)),
     sg.Input('1', size=(12,1), key='root1_lower'),
     sg.Input('10', size=(12,1), key='root1_upper')],

    [sg.Text('Tip Chord 1', size=(20,1)),
     sg.Input('1', size=(12,1), key='tip1_lower'),
     sg.Input('10', size=(12,1), key='tip1_upper')],

    [sg.Text('Semi Span 1', size=(20,1)),
     sg.Input('1', size=(12,1), key='span1_lower'),
     sg.Input('20', size=(12,1), key='span1_upper')],

    [sg.Text('Root Thickness 1', size=(20,1)),
     sg.Input('0.1', size=(12,1), key='rootth1_lower'),
     sg.Input('1', size=(12,1), key='rootth1_upper')],

    [sg.Text('Tip Thickness 1', size=(20,1)),
     sg.Input('0.1', size=(12,1), key='tipth1_lower'),
     sg.Input('1', size=(12,1), key='tipth1_upper')],

    [sg.Text('Mach', size=(20,1)),
     sg.Input('0.5', size=(12,1), key='mach_lower'),
     sg.Input('5', size=(12,1), key='mach_upper')],

    [sg.Text('Alpha', size=(20,1)),
     sg.Input('0', size=(12,1), key='alpha_lower'),
     sg.Input('15', size=(12,1), key='alpha_upper')],

    [sg.Text('Altitude', size=(20,1)),
     sg.Input('0', size=(12,1), key='alt_lower'),
     sg.Input('30000', size=(12,1), key='alt_upper')],

    [sg.HorizontalSeparator()],

    [sg.Text('Optimization Settings',
             font=('Any', 14))],

    [sg.Text('Population Size'),
     sg.Input('10', key='popsize')],

    [sg.Text('Max Iterations'),
     sg.Input('20', key='maxiter')],

    [
        sg.Button('Run Optimization'),
        sg.Button('Clear Optimization')
    ],

    [
        sg.Multiline(
        size=(120,30),
        key='opt_output',
        expand_x=True,
        expand_y=True,
        autoscroll=True
        )
    ]
]

# =========================================================
# TAB 3 : FLIGHT ENVELOPE
# =========================================================
flight_tab = [
    [sg.Text('Alpha Sweep', font=('Any', 14))],

    [
        sg.Text('Min'),
        sg.Input('0', size=(5,1), key='alpha_min'),

        sg.Text('Max'),
        sg.Input('10', size=(5,1), key='alpha_max'),

        sg.Text('Step'),
        sg.Input('1', size=(5,1), key='alpha_step')
    ],

    [sg.HorizontalSeparator()],

    [sg.Text('Mach Sweep', font=('Any', 14))],

    [
        sg.Text('Min'),
        sg.Input('0.5', size=(5,1), key='mach_min'),

        sg.Text('Max'),
        sg.Input('3', size=(5,1), key='mach_max'),

        sg.Text('Step'),
        sg.Input('0.5', size=(5,1), key='mach_step')
    ],

    [sg.HorizontalSeparator()],

    [
        sg.Button('Run Flight Envelope Analysis'),
        sg.Button('Clear Analysis')
    ],

    [
        sg.Multiline(
            size=(120, 30),
            key='flight_output',
            expand_x=True,
            expand_y=True,
            autoscroll=True
        )
    ]
]
# =========================================================
# MAIN LAYOUT
# =========================================================

layout = [

    [
        sg.Text(
            'Optimization Aerodynamic Configuration Design of Aerospace Vehicles',
            font=('Any', 16)
        )
    ],

    [
        sg.TabGroup([
            [

                sg.Tab('Simple Prediction',
                       prediction_tab),

                sg.Tab('Optimization',
                       optimization_tab),

                sg.Tab('Flight Envelope',
                       flight_tab)

            ]
        ])
    ]
]


# =========================================================
# WINDOW
# =========================================================
window = sg.Window(
    'Aerospace Design GUI',
    layout,
    size=(1400, 900),
    resizable=True,
    finalize=True
)

window.maximize()

# =========================================================
# EVENT LOOP
# =========================================================

while True:

    event, values = window.read()

    if event in (sg.WINDOW_CLOSED, 'Exit'):
        break


    # =====================================================
    # SIMPLE PREDICTION
    # =====================================================
    if event == 'Clear':
        for key in values:
            try:
                window[key].update('')
            except:
                pass
    if event == 'Estimate':

        params = {

            'nose_len': values['nose_len'],
            'body_len': values['body_len'],
            'wing_le': values['wing_le'],
            'root_chord': values['root_chord'],
            'tip_chord': values['tip_chord'],
            'semi_span': values['semi_span'],
            'root_th': values['root_th'],
            'tip_th': values['tip_th'],
            'wing_sweep': values['wing_sweep'],

            'tail_le': values['tail_le'],
            'root_chord1': values['root_chord1'],
            'tip_chord1': values['tip_chord1'],
            'semi_span1': values['semi_span1'],
            'root_th1': values['root_th1'],
            'tip_th1': values['tip_th1'],

            'mach': values['mach'],
            'alpha': values['alpha'],
            'alt': values['alt']
        }

        result = aerodynamic_prediction(params)

        window['cl_out'].update(result['CL'])
        window['cd_out'].update(result['CD'])
        window['xcp_out'].update(result['XCP'])


    # =====================================================
    # OPTIMIZATION
    # =====================================================

    if event == 'Run Optimization':

        bounds = [

            (float(values['nose_lower']),
             float(values['nose_upper'])),

            (float(values['body_lower']),
             float(values['body_upper'])),

            (float(values['wingle_lower']),
             float(values['wingle_upper'])),

            (float(values['root_lower']),
             float(values['root_upper'])),

            (float(values['tip_lower']),
             float(values['tip_upper'])),

            (float(values['span_lower']),
             float(values['span_upper'])),

            (float(values['rootth_lower']),
             float(values['rootth_upper'])),

            (float(values['tipth_lower']),
             float(values['tipth_upper'])),

            (float(values['sweep_lower']),
             float(values['sweep_upper'])),

            (float(values['tail_lower']),
             float(values['tail_upper'])),

            (float(values['root1_lower']),
             float(values['root1_upper'])),

            (float(values['tip1_lower']),
             float(values['tip1_upper'])),

            (float(values['span1_lower']),
             float(values['span1_upper'])),

            (float(values['rootth1_lower']),
             float(values['rootth1_upper'])),

            (float(values['tipth1_lower']),
             float(values['tipth1_upper'])),

            (float(values['mach_lower']),
             float(values['mach_upper'])),

            (float(values['alpha_lower']),
             float(values['alpha_upper'])),

            (float(values['alt_lower']),
             float(values['alt_upper']))
        ]

        result = run_optimization(
            bounds,
            int(values['maxiter']),
            int(values['popsize'])
        )

        output_text = (
            f'Optimization Completed\n\n'
            f'Best Objective (CL/CD) = {-result.fun}\n\n'
            f'Best Parameters:\n\n'
            f'{result.x}'
        )

        window['opt_output'].update(output_text)


    # =====================================================
    # FLIGHT ENVELOPE
    # =====================================================

    if event == 'Run Flight Envelope Analysis':

        params = {

            'nose_len': values['nose_len'],
            'body_len': values['body_len'],
            'wing_le': values['wing_le'],
            'root_chord': values['root_chord'],
            'tip_chord': values['tip_chord'],
            'semi_span': values['semi_span'],
            'root_th': values['root_th'],
            'tip_th': values['tip_th'],
            'wing_sweep': values['wing_sweep'],

            'tail_le': values['tail_le'],
            'root_chord1': values['root_chord1'],
            'tip_chord1': values['tip_chord1'],
            'semi_span1': values['semi_span1'],
            'root_th1': values['root_th1'],
            'tip_th1': values['tip_th1'],

            'mach': values['mach'],
            'alpha': values['alpha'],
            'alt': values['alt']
        }

        alpha_results = alpha_sweep(
            params,
            float(values['alpha_min']),
            float(values['alpha_max']),
            float(values['alpha_step'])
        )

        mach_results = mach_sweep(
            params,
            float(values['mach_min']),
            float(values['mach_max']),
            float(values['mach_step'])
        )

        final_text = 'ALPHA SWEEP RESULTS\n\n'

        for r in alpha_results:
            final_text += r + '\n'

        final_text += '\n\nMACH SWEEP RESULTS\n\n'

        for r in mach_results:
            final_text += r + '\n'

        window['flight_output'].update(final_text)

window.close()