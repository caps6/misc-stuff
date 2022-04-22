# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn theme.
sns.set_theme(style = 'darkgrid')

history = np.zeros((sample_size, n_items), dtype = int)

# Sampling parameters.
n_items = 1000
sample_size = 10

sample = np.zeros(sample_size, dtype = int)
steps = np.arange(n_items)

# Initial sample.
for nn in range(sample_size):
    sample[nn] = nn

for nn in range(sample_size, n_items):

    # Compute the current item.
    ii = np.random.randint(0, nn, dtype = int)

    if ii < sample_size:
        sample[ii] = nn

    # Keep track of sample evolution.
    history[:, nn] = sample

fig, ax = plt.subplots(figsize = (14, 8))
for ii in range(sample_size):
    ax.plot(steps, history[ii, :], label = f'{ii}-th point')
ax.grid(which = 'major', axis = 'both', linestyle='--')
ax.set_title(f'Sampling with N={n_items}, k={sample_size}')
ax.set_xlabel('Items')
ax.set_ylabel('Sample elements')
plt.legend(loc = 'best')
fig.savefig('sampled.png')
