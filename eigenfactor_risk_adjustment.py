import numpy as np
from numpy import linalg as la


def simulation_factor_return(d0_diag, u0, t):
    # Simulation factor return
    bm = np.array([np.random.normal(0, dk, t) for dk in d0_diag])
    return np.dot(u0, bm)


def calculate_simulated_volatility(fm, origin_matrix):
    Fm = np.cov(fm)
    dm_diag, um = la.eig(Fm)
    dm = np.diag(dm_diag)
    dm_til = np.dot(np.dot(um.T, origin_matrix), um)
    v = np.sqrt((dm / dm_til).mean(axis=1))
    return v


def scale_volatility(v, a=1.4):
    # TODO 检查vk的生成逻辑
    vs = a * (v - 1) + 1
    return vs


def adjust_covariance_matrix(d0, v, u0):
    # Firstly, we get the adjustment eigenmatrix
    # And then use it to get the final adjustment covariance matrix
    d0_til = np.dot(np.diag(np.square(v)), d0)
    return np.dot(np.dot(u0, d0_til), u0.T)


if __name__ == '__main__':
    origin = np.array([[1., 1., 0.], [1., 2., 0.], [0., 0., 3.]])
    # Compute the eigenvalues and right eigenvectors of the origin matrix.
    # u0 is the right eigenvectors of origin_matrix
    # Caution: d0_diag is a vector of eigenvalues no a matrix!
    d0_diag, u0 = la.eig(origin)
    fm = simulation_factor_return(d0_diag, u0, t=10000)
    v = calculate_simulated_volatility(fm, origin)
    vs = scale_volatility(v)
    f0_til = adjust_covariance_matrix(np.diag(d0_diag), vs, u0)
    print(f0_til)
