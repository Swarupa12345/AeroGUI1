from predictor import aerodynamic_prediction


def alpha_sweep(params, amin, amax, astep):

    results = []

    alpha = amin

    while alpha <= amax:

        params['alpha'] = alpha

        result = aerodynamic_prediction(params)

        text = (
            f'Alpha={alpha}  '
            f'CL={result["CL"]}  '
            f'CD={result["CD"]}'
        )

        results.append(text)

        alpha += astep

    return results


def mach_sweep(params, mmin, mmax, mstep):

    results = []

    mach = mmin

    while mach <= mmax:

        params['mach'] = mach

        result = aerodynamic_prediction(params)

        text = (
            f'Mach={mach}  '
            f'CL={result["CL"]}  '
            f'CD={result["CD"]}'
        )

        results.append(text)

        mach += mstep

    return results