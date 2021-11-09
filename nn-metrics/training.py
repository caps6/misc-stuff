#! /usr/bin/env python3
"""
Focal loss: a case study with credit card fraud detection.
Credits to:
    - https://www.kaggle.com/mlg-ulb/creditcardfraud/code
    - https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/
    - https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import TruePositives, FalseNegatives, Recall
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from prettytable import PrettyTable
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#
# PARAMETERS
#
np.random.seed(42)
#mpl.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Set seaborn theme.
sns.set_theme(style='darkgrid')

EPOCHS = 5
BATCH_SIZE = 2048

#
# DATA LOADING/CLEANSING
#

# Load data from CSV file.
df_raw = pd.read_csv('creditcardfraud/creditcard.csv')
n_samples = len(df_raw)
print(f'Num. of samples: {n_samples}.')

# Check size of samples.
df_pos = df_raw[df_raw['Class'] == 1]
n_pos_samples = len(df_pos)
pos_ratio = 100 * n_pos_samples / n_samples
print(f'Num. of positive samples: {n_pos_samples} ({pos_ratio:.2f}% of total).')

# Drop useless data and convert amount to log space.
df_cleaned = df_raw.copy()
df_cleaned.pop('Time')
df_cleaned['log-amount'] = np.log(df_cleaned.pop('Amount') + 0.001)

# Double train/test split for testing and validation data.
df_train, df_test = train_test_split(df_cleaned, test_size = 0.2, shuffle = True)
df_train, df_valid = train_test_split(df_train, test_size = 0.2, shuffle = True)

print(f'Size of training data: {len(df_train)}.')
print(f'Size of validation data: {len(df_valid)}.')
print(f'Size of test data: {len(df_test)}.')

# Extract labels and features from data.
labels_train = np.array(df_train.pop('Class'))
labels_valid = np.array(df_valid.pop('Class'))
labels_test = np.array(df_test.pop('Class'))
features_train = np.array(df_train)
features_valid = np.array(df_valid)
features_test = np.array(df_test)

# Normalize data.
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_valid = scaler.transform(features_valid)
features_test = scaler.transform(features_test)

# Enforce lower/upper bounds.
features_train = np.clip(features_train, -5, 5)
features_valid = np.clip(features_valid, -5, 5)
features_test = np.clip(features_test, -5, 5)
n_features = features_train.shape[-1]

#
# MODEL TRAINING
#

# Model parameters.
opt = Adam(learning_rate = 1e-3)

metrics = [
    TruePositives(name = 'tp'),
    FalseNegatives(name = 'fn'),
    Recall(name = 'recall')
]

losses = [
    BinaryCrossentropy(),
    SigmoidFocalCrossEntropy(gamma = 2, alpha = 4)
]

loss_names = [
    'binary cross-entropy',
    'focal loss'
]

logs_loss = []
logs_recall = []

for loss in losses:

    # Setup/compile the model.
    model = Sequential()
    model.add(Dense(16, input_dim = n_features, activation = 'relu',
        kernel_initializer = 'he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = opt, loss = loss, metrics = metrics)

    # Fit the model.
    logs = model.fit(features_train, labels_train, validation_data = (features_valid,
        labels_valid), epochs = EPOCHS, verbose = 0)

    logs_loss.append(logs.history['loss'])
    logs_recall.append(logs.history['recall'])

    # Evaluate the model.
    eval_train = model.evaluate(features_train, labels_train, verbose = 0)
    eval_test = model.evaluate(features_valid, labels_valid, verbose = 0)

    table = PrettyTable()
    table.field_names = ['Data', 'Loss', 'TruePositives', 'FalseNegatives', 'Recall']

    for stage, eval_info in zip(('training', 'test'), (eval_train, eval_test)):

        row = [stage]

        for ii, lbl in enumerate(model.metrics_names):
            row.append(f'{eval_info[ii]:.3f}')

        table.add_row(row)

    print('\n')
    print(table)

#
# Plotting
#

fig, ax = plt.subplots()
for log_loss, name in zip(logs_loss, loss_names):
    ax.plot(log_loss, label = name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss values with different loss functions')
fig.savefig(f'losses.png')

fig, ax = plt.subplots()
for log_recall, name in zip(logs_recall, loss_names):
    ax.plot(log_recall, label = name)
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()
plt.title('Evaluation metric with different loss functions')
fig.savefig(f'accuracies.png')
