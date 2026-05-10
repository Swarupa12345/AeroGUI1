import math


def aerodynamic_prediction(params):

    nose_len = float(params['nose_len'])
    body_len = float(params['body_len'])
    wing_le = float(params['wing_le'])
    root_chord = float(params['root_chord'])
    tip_chord = float(params['tip_chord'])
    semi_span = float(params['semi_span'])
    root_th = float(params['root_th'])
    tip_th = float(params['tip_th'])
    wing_sweep = float(params['wing_sweep'])

    tail_le = float(params['tail_le'])
    root_chord1 = float(params['root_chord1'])
    tip_chord1 = float(params['tip_chord1'])
    semi_span1 = float(params['semi_span1'])
    root_th1 = float(params['root_th1'])
    tip_th1 = float(params['tip_th1'])

    mach = float(params['mach'])
    alpha = float(params['alpha'])
    alt = float(params['alt'])

    alpha_rad = math.radians(alpha)

    # Wing Area
    wing_area = ((root_chord + tip_chord) / 2.0) * semi_span * 2

    # Tail Area
    tail_area = ((root_chord1 + tip_chord1) / 2.0) * semi_span1 * 2

    # Total Area
    total_area = wing_area + tail_area

    # Thickness Ratio
    thickness_ratio = ((root_th + tip_th) / 2.0) / root_chord

    # Lift Coefficient
    cl = (2 * math.pi * alpha_rad) * (
        1 / math.sqrt(abs(1 - mach**2) + 0.01)
    )

    cl = cl * (1 + 0.02 * wing_sweep / 45)

    # Drag Coefficient
    cd0 = 0.02 + 0.002 * thickness_ratio * 100

    induced_drag = (cl ** 2) / (math.pi * 4 * 0.85)

    wave_drag = 0.0

    if mach > 1:
        wave_drag = 0.08 * (mach - 1) ** 2

    cd = cd0 + induced_drag + wave_drag

    # Center of Pressure
    xcp = (
        0.4 * nose_len +
        0.35 * body_len +
        0.25 * wing_le
    )

    return {
        'CL': round(cl, 4),
        'CD': round(cd, 4),
        'XCP': round(xcp, 4)
    }