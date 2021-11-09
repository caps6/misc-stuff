#!/usr/bin/env python3
"""Plotting binary entropy."""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Set seaborn theme.
sns.set_theme(style='darkgrid')

def binary_entropy(p):
    """Entropy of a binary random variable."""

    ent = - (p * np.log2(p) + (1-p) * np.log2(1-p))

    return ent

# Entropy values.
probs = np.linspace(0.001, 0.999, num = 100)
entropies = [binary_entropy(p) for p in probs]

#
# Plotting.
#
fig, ax = plt.subplots()
ax.plot(probs, entropies)
ax.annotate('max uncertainty', xy = (155, 240), xytext = (40, 250),
    xycoords = 'axes points', arrowprops = dict(facecolor = 'red'))
ax.annotate('min uncertainty', xy = (20, 10), xytext = (130, 30),
    xycoords = 'axes points', arrowprops = dict(facecolor = 'black'))
ax.annotate('min uncertainty', xy = (330, 10), xytext = (130, 30),
    xycoords = 'axes points', arrowprops = dict(facecolor = 'black'))
plt.ylim([0, 1.15])
plt.xlabel('p')
plt.ylabel('H(x)')
plt.title('Entropy of a Bernoulli variable x with parameter p')
fig.savefig('bernoulli_entropy.png')
