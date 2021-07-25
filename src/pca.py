import numpy as np

# calculate mean
def mean(x):
    length = len(x)
    m = sum(x) / length

    return m

# covariance matrix
def covariance(x):
    X = x - mean(x)
    cov = np.cov(X.T)

    return cov

# obtain eigenvalue and eigenvectors
def Eig(x, num_components):
    cov = covariance(x)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    # sort eigenvectors according to eigen values
    eigenvectors = eigenvectors.T
    idxs = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idxs]
    eigenvectors = eigenvectors[idxs]

    components = eigenvectors[0:num_components]

    return components


def transform(x):
    X = x - mean(x)
    com = Eig(x, 2)
    return np.dot(X, com.T)



