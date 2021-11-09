#!/usr/bin/env python3
"""Plotting different loss functions used in DL models."""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Set seaborn theme.
sns.set_theme(style='darkgrid')

def cross_entropy(y_true, y_pred):

    if y_true == 1:
        loss = -np.log(y_pred)
    else:
        loss = -np.log(1 - y_pred)

    return loss

def hinge(y_true, y_pred):
    return max(0, y_true - (1 - 2 * y_true) * y_pred)

def L1(y_true, y_pred):
    return np.absolute(y_pred - y_true)

def focal_loss(y_true, y_pred, alpha):

    if y_true == 1:
        loss = - ((1- y_pred) ** alpha) * np.log(y_pred)
    else:
        loss = - (y_pred ** alpha) * np.log(1 - y_pred)

    return loss

y_pred = np.linspace(0.00001, 1, num = 100)

# Evaluate losses.
loss_ce = [cross_entropy(1, y) for y in y_pred]
#loss_hi = [hinge(1, y) for y in y_pred]
loss_l1 = [L1(1, y) for y in y_pred]

loss_fl_1 = [focal_loss(1, y, 1) for y in y_pred]
loss_fl_2 = [focal_loss(1, y, 2) for y in y_pred]
loss_fl_3 = [focal_loss(1, y, 3) for y in y_pred]


#
# Plot the timeline of populations.
#
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots()
plt.plot(y_pred, loss_ce, label = 'Cross Entropy')
#plt.plot(y_pred, loss_hi, label = 'Hinge')
plt.plot(y_pred, loss_fl_1, label = 'Focal Loss with lambda=1')
plt.plot(y_pred, loss_fl_2, label = 'Focal Loss with lambda=2')
plt.plot(y_pred, loss_fl_3, label = 'Focal Loss with lambda=3')
#plt.plot(y_pred, loss_l1, label = 'L1')
plt.grid(True)
plt.legend(loc = 'best')
plt.xlabel('Estimated probability yp')
plt.ylabel('Loss value')
#plt.title('Binary cross-entropy with class label yt=1')
plt.title('Loss functions for binary classification, with class label yt=1')
fig.savefig('losses.png')
