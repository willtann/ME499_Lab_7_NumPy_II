
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def eggholder(x, y):
    """
    Takes in either a scalar x and y, or a 1-D numpy array for both x and y.
    """
    if (isinstance(x, np.ndarray) and not isinstance(y, np.ndarray)) or (
            isinstance(y, np.ndarray) and not isinstance(x, np.ndarray)):
        pass

    result = -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))
    if isinstance(x, np.ndarray):
        outside = (np.abs(x) > 512) | (np.abs(y) > 512)
        result[outside] = 2000
        return result
    else:
        return 2000 if (abs(x) > 512 or abs(y) > 512) else result


def minimize_eggholder(guess, max_calls=100):
    """
    :param guess: scalar x, y array of any length and uses these to initialize an optimization function
    :param max_calls: Maximum number of calls for optimization function
    :return: settled on value of optimization function of same length as guess
    """
    # function to optimize
    egg = lambda xy: eggholder(xy[0], xy[1])
    mini = optimize.fmin(func=egg, x0=guess, maxfun=max_calls, full_output=True)
    # save results
    return mini[0], mini[1]


if __name__ == '__main__':
    guess = np.random.randint(-512, 512, size=(1000, 2))

    # fmin
    my_test = minimize_eggholder(guess)
    # function values from optimized function
    f_min = my_test[1]

    # gmin
    g_min = [None] * len(guess)
    # function values from global
    for i, loc in enumerate(guess):
        global_min = eggholder(loc[0], loc[1])
        g_min[i] = global_min
    np.array(g_min).reshape((len(guess)), 1)

    # diff
    diff = abs(f_min - g_min)

    # Plotting
    plt.hist(diff, bins=25)
    plt.title('Histogram: Absolute difference')
    plt.ylabel('Frequency')
    plt.xlabel('Absolute difference (fmin - global min)')
    plt.savefig('Histogram')
    plt.show()
