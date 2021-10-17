from random import choice
from time import time
from numba import jit
from numba.typed import List
import matplotlib.pyplot as plt
import seaborn as sns

def translate_list_01(char_list):
    """Working on a list.
    Pure Python implementation."""

    num_list = []
    for word in char_list:

        if word == 'A':
            num = 1
        elif word == 'B':
            num = 2
        elif word == 'C':
            num = 3
        elif word == 'D':
            num = 4
        else:
            num = 5

        num_list.append(num)

    return num_list

@jit(nopython=True)
def translate_list_02(char_list):
    """Working on a list.
    CPU-acceleration with Numba."""

    num_list = []
    for word in char_list:

        if word == 'A':
            num = 1
        elif word == 'B':
            num = 2
        elif word == 'C':
            num = 3
        elif word == 'D':
            num = 4
        else:
            num = 5

        num_list.append(num)

    return num_list


if __name__ == '__main__':

    logN = [ee for ee in range(17, 25)]
    NN = [2**ee for ee in logN]

    python_times = []
    numba_times = []

    idx = 0

    for N in NN:

        print(f'Running with {N}')

        # Prepare lists.
        list_01 = [choice('ABCDE') for _ in range(N)]
        list_02 = List(list_01)

        # Pure python method.
        t_a = time()
        num_list = translate_list_01(list_01)
        t_b = time()
        python_times.append(t_b - t_a)

        # CPU-accelerated Numba method.
        t_a = time()
        num_list = translate_list_02(list_02)
        t_b = time()
        numba_times.append(t_b - t_a)

    # To ignore warm-up time.
    logN = logN[1:]

    # Plot time bars.
    x1 = [x - 0.125 for x in logN]
    x2 = [x + 0.25 for x in logN]

    sns.set_style('darkgrid')

    fig, ax = plt.subplots()
    ax.bar(logN, python_times[1:], label = 'Pure python', width = 0.25)
    ax.bar(x2, numba_times[1:], label = 'Numba-CPU', width = 0.25)
    ax.set(xlabel = 'logN (list size: 2^logN)', ylabel = 'Time [s]',
        title = 'Execution time')
    ax.set_xticks(logN)
    ax.grid(True)
    plt.legend()
    fig.savefig('char_times.png')
    plt.close()

    # Time gains
    gains = [x / (y +  0.0000001) for x, y in zip(python_times, numba_times)]

    fig, ax = plt.subplots()
    ax.bar(logN, gains[1:], width = 0.25)
    ax.set(xlabel = 'logN (list size: 2^logN)',
        ylabel = 'Gain factor',
        title = 'Speed-up achieved with Numba-CPU')
    ax.grid(True)
    fig.savefig('char_gains.png')
    plt.close()
