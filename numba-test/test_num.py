import math
from time import time, sleep
from numba import jit, cuda
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def computing_01(A, B, a, b):
    """Numpy operations with no acceleration."""

    # Scalar multiplication and addition of matrices.
    Y = a * A + b * B
    # Scalar multiplication and subtraction of matrices.
    Y -= a * A - b * B
    # Element-wise logarithm.
    Y += np.log(A) + np.log(B)
    # Element-wise exponential.
    Y -= np.exp(A) - np.exp(B)
    # Element-wise minimum and maximum.
    Y += (np.maximum(A, B) - np.minimum(A, B)) / 2
    # Element-wise multiplication and division.
    Y -= np.multiply(A, B) - np.divide(A, B)

    return Y

@jit(nopython=True)
def computing_02(A, B, a, b):
    """Numpy operations with Numba acceleration."""

    # Scalar multiplication and addition of matrices.
    Y = a * A + b * B
    # Scalar multiplication and subtraction of matrices.
    Y -= a * A - b * B
    # Element-wise logarithm.
    Y += np.log(A) + np.log(B)
    # Element-wise exponential.
    Y -= np.exp(A) - np.exp(B)
    # Element-wise minimum and maximum.
    Y += (np.maximum(A, B) - np.minimum(A, B)) / 2
    # Element-wise multiplication and division.
    Y -= np.multiply(A, B) - np.divide(A, B)

    return Y

@jit(nopython=True)
def computing_03(A, B, a, b):
    """Numba acceleration, without Numpy."""

    # Matrix size.
    N = A.shape[0]

    # Init temporary matrices.
    Y = np.empty((N,N))

    for ii in range(N):
        for jj in range(N):

            # Scalar multiplication and addition of matrices.
            Y[ii, jj] = a * A[ii, jj] + b * B[ii, jj]
            # Scalar multiplication and subtraction of matrices.
            Y[ii, jj] -= a * A[ii, jj] - b * B[ii, jj]
            # Element-wise logarithm.
            Y[ii, jj] += math.log(A[ii, jj]) + math.log(B[ii, jj])
            # Element-wise exponential.
            Y[ii, jj] += math.exp(A[ii, jj]) - math.exp(B[ii, jj])
            # Element-wise minimum and maximum.
            Y[ii, jj] += (max(A[ii, jj], B[ii, jj]) - min(A[ii, jj], B[ii, jj])) / 2
            # Element-wise multiplication and division.
            Y[ii, jj] -= A[ii, jj] * B[ii, jj] - A[ii, jj] / B[ii, jj]

    return Y

@cuda.jit
def computing_04(Y, A, B, a, b, N, size):
    """Operations accelerated with Cuda."""

    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw

    if pos < size:

        ii = int(math.floor( pos / N ))
        jj = int(math.floor( (pos - ii * N) / N ))

        # Scalar multiplication and addition of matrices.
        Y[ii, jj] = a * A[ii, jj] + b * B[ii, jj]
        # Scalar multiplication and subtraction of matrices.
        Y[ii, jj] -= a * A[ii, jj] - b * B[ii, jj]
        # Element-wise logarithm.
        Y[ii, jj] += math.log(A[ii, jj]) + math.log(B[ii, jj])
        # Element-wise exponential.
        Y[ii, jj] += math.exp(A[ii, jj]) - math.exp(B[ii, jj])
        # Element-wise minimum and maximum.
        Y[ii, jj] += (max(A[ii, jj], B[ii, jj]) - min(A[ii, jj], B[ii, jj])) / 2
        # Element-wise multiplication and division.
        Y[ii, jj] -= A[ii, jj] * B[ii, jj] - A[ii, jj] / B[ii, jj]


if __name__ == '__main__':

    NN = np.array([1024 * ee for ee in range(1, 11)])
    rng = np.random.default_rng(2021)

    numpy_times = np.zeros(len(NN))
    numba_numpy_times = np.zeros(len(NN))
    numba_times = np.zeros(len(NN))
    cuda_times = np.zeros(len(NN))

    idx = 0

    a = rng.random()
    b = rng.random()

    for N in NN:

        print(f'Running with {N}')

        A = rng.random((N, N))
        B = rng.random((N, N))

        t_a = time()
        Y = computing_01(A, B, a, b)
        t_b = time()
        numpy_times[idx] = t_b - t_a

        t_a = time()
        Y = computing_02(A, B, a, b)
        t_b = time()
        numba_numpy_times[idx] = t_b - t_a

        t_a = time()
        Y = computing_03(A, B, a, b)
        t_b = time()
        numba_times[idx] = t_b - t_a

        size = N * N
        threadsperblock = 256
        blockspergrid = np.ceil(size / threadsperblock).astype('int')

        # CUDA function.
        t_a = time()

        # Transfer input/output arrays to GPU.
        A = cuda.to_device(A)
        B = cuda.to_device(B)
        Y = cuda.to_device(np.zeros((N,N)))

        computing_04[blockspergrid, threadsperblock](Y, A, B, a, b, N, size)

        # Get back output array from GPU.
        Y.copy_to_host()

        t_b = time()
        cuda_times[idx] = t_b - t_a

        idx += 1

    sns.set_style("darkgrid")

    fig, ax = plt.subplots()
    w = 200
    ax.bar(NN[1:] - 3/2*w, numpy_times[1:], label = 'Numpy only', width = w)
    ax.bar(NN[1:] - w/2, numba_numpy_times[1:], label = 'Numba + Numpy', width = w)
    ax.bar(NN[1:] + w/2, numba_times[1:], label = 'Numba-CPU', width = w)
    #ax.bar(NN[1:] + 3/2*w, cuda_times[1:], label = 'Numba-CUDA', width = w)
    ax.set(xlabel = 'N', ylabel = 'Time (s)', title = 'Execution time')
    ax.grid(True)
    plt.legend()
    fig.savefig('num_times.png')
    plt.close()
