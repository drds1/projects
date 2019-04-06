import statsmodels.formula.api as smf

def forward_selected(data, response):
    """Linear model designed by forward selection.
    DO NOT PUT SPACES IN data DATAFRAME COLUMN NAMES OR RESPONSE argument
    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import time
    # fake data
    n = 2000
    k = 100
    ptrue = [20.0, 13.0]
    t = np.arange(1, n + 1)
    # make fake datsa identical to fortran
    y = 3.2 * np.sin(2 * 3.1415926535 / ptrue[0] * t) + 8.0 * np.sin(2 * 3.1415926535 / ptrue[1] * t)
    X = np.zeros((n, k))
    for ik in range(k):
        X[:, ik] = np.random.randn(n)

    for ik in range(len(ptrue)):
        X[:, ik] = np.sin(2 * 3.1415926535 / ptrue[ik] * t)

    response = ['driver']+['column_'+np.str(i) for i in range(k)]
    yv = np.zeros((n,1))
    yv[:,0] = y
    combine = np.hstack((yv,X))
    xdf = pd.DataFrame(combine,columns = response)

    ts1 = time.time()
    model = forward_selected(xdf,'driver')
    ts2 = time.time()
    print('greedy search time',ts2-ts1)