#
# File: project1.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
        To import from another file xyz.py here, type
        import project1_py.xyz
        However, do not import any modules except numpy in those files.
        It's ok to import modules only in files that are
        not imported here (e.g. for your plotting code).
'''
import numpy as np

alpha_table = {'simple1': 0.00056, 'simple2': 0.01, 'simple3': 0.001, 'secret1': 0.001, 'secret2': 0.001}

def optimize(f, g, x0, n, count, prob):
    """
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `g` costs twice of `f`
        count (function): takes no arguments are returns current count
        prob (str): Name of the problem. So you can use a different strategy
                 for each problem. `prob` can be `simple1`,`simple2`,`simple3`,
                 `secret1` or `secret2`
    Returns:
        x_best (np.array): best selection of variables found
    """
    x_hist = [x0]
    #if prob == 'simple1' or prob == 'simple2' or prob == 'simple3':
    x_hist = gradient_descent(f,g,x0,n,count, prob)
    return x_hist[-1]


def gradient_descent(f,g,x0,n,count, prob):
    alpha = alpha_table[prob]
    x_hist = [x0]
    while count() < n:
        x_hist.append(x_hist[-1] - alpha * g(x_hist[-1]))
    return x_hist