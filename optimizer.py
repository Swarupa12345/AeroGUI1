from scipy.optimize import differential_evolution

from predictor import aerodynamic_prediction


def objective_function(x):

    params = {

        'nose_len': x[0],
        'body_len': x[1],
        'wing_le': x[2],
        'root_chord': x[3],
        'tip_chord': x[4],
        'semi_span': x[5],
        'root_th': x[6],
        'tip_th': x[7],
        'wing_sweep': x[8],

        'tail_le': x[9],
        'root_chord1': x[10],
        'tip_chord1': x[11],
        'semi_span1': x[12],
        'root_th1': x[13],
        'tip_th1': x[14],

        'mach': x[15],
        'alpha': x[16],
        'alt': x[17]
    }

    result = aerodynamic_prediction(params)

    cl = result['CL']
    cd = result['CD']

    objective = cl / cd

    return -objective


def run_optimization(bounds, maxiter, popsize):

    result = differential_evolution(
        objective_function,
        bounds,
        maxiter=maxiter,
        popsize=popsize
    )

    return result