#! /Users/tannerwilliams/Desktop/ME499/ME499_Lab_7_NumPy_II/inertia.py
import numpy as np
import matplotlib.pyplot as plt
import scipy
import random

"""
References:
    [1]: https://scipython.com/book/chapter-6-numpy/problems/p65/the-moment-of-inertia-tensor/
"""


def compute_inertia_matrix(n_x_three_array, mass=1):
    """
    This function takes in an array and returns the inertia matrix for that object
    :param n_x_three_array: Nx3 array of x, y, z points in space
    :param mass: Mass of entire object in question
    :return: Inertia matrix based on sampled points
    """
    # point mass from total mass
    m = mass/len(n_x_three_array)
    # Convert input array to numpy array and make 1XN array of values for each coordinate type
    x = np.array(n_x_three_array)[:, 0].T
    y = np.array(n_x_three_array)[:, 1].T
    z = np.array(n_x_three_array)[:, 2].T
    # Return sum of inertia matrix for all sampled points
    return np.array([[np.sum(m * (y ** 2 + z ** 2)), -np.sum(m * x * y), -np.sum(m * x * z)],
                     [-np.sum(m * x * y), np.sum(m * (x ** 2 + z ** 2)), -np.sum(m * y * z)],
                     [-np.sum(m * x * z), -np.sum(m * y * z), np.sum(m * (x ** 2 + y ** 2))]])


def sample_sphere_polar(N, r=1):
    """
    Use random number drawing of angle to generate random points on a sphere
    :param N: Number of random points
    :param r: Radius of sphere defaulting to 1
    :return: Random points on spheres surface
    """
    # Array for holding output
    points = np.reshape([None] * (3 * N), (N, 3))
    # Generate random angles for each point to reference
    for i in range(N):
        phi = float(np.random.randint(0, 180) * (np.pi/180))  # [0, pi]
        theta = float(np.random.randint(0, 360) * (np.pi/180))  # [0, 2*pi]
        # Store each random points coordinates
        points[i] = (r * np.sin(phi) * np.cos(theta)), (r * np.sin(phi) * np.cos(theta)), (r * np.cos(phi))
        # Return Nx3 array of each random point
    return points


def sample_sphere_gaussian(N, r=1):
    """
    Use random gaussian sampling method to generate random points on a sphere of given radius
    :param N: Number of random points
    :param r: Radius of sphere defaulting to 1
    :return: Random points on spheres surface
    """
    raw_samples = np.random.standard_normal(size=(N, 3))
    a = np.array([None] * N)
    b = np.array([None] * N)
    for i in range(N):
        a[i] = r / np.linalg.norm(raw_samples[i])
        raw_samples[i] = a[i] * raw_samples[i]
    return raw_samples


def test_inertia_matrices_output():
    np.set_printoptions(precision=3, suppress=True)
    m = 1
    r = 1
    n = 1000

    """ Polar Interia Matrix"""
    polar = compute_inertia_matrix(sample_sphere_polar(n))
    print('Polar: ')
    print(polar)

    """ Gaussian Interia Matrix"""
    gauss = compute_inertia_matrix(sample_sphere_gaussian(n))
    print('Gaussian: ')
    print(gauss)

    """ Expected Inertia matrix"""
    expected = np.array([[(0.5 * m * r ** 2), 0, 0],
                         [0, (0.5 * m * r ** 2), 0],
                         [0, 0, (0.5 * m * r ** 2)]])
    print('Expected: ')
    print(expected)


# if __name__ == '__main__':
#     print('-----Problem 1.1-----')
#     print(compute_inertia_matrix([[1, 2, 3], [4, 5, 6]]))
#
#     print('-----Problem 1.2-----')
#     print(sample_sphere_polar(2))
#
#     print('-----Problem 1.3-----')
#     test_gauss = sample_sphere_gaussian(2)
#     print(test_gauss)
#     print('r[0] = ', np.linalg.norm(test_gauss[0]))
#     print('r[1] =',  np.linalg.norm(test_gauss[1]))
#
#     print('-----Problem 1.4-----')
#     print(test_inertia_matrices_output())
