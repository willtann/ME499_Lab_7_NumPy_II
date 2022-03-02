import numpy as np
from scipy import optimize


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
    for i, points in enumerate(guess):
        g = guess[i]


if __name__ == '__main__':
    rand_points = np.random.randint(-512, 512, size=(1000, 2))
    test_min = minimize_eggholder(rand_points)
