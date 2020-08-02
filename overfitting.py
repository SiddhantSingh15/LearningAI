import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.python.keras.callbacks import TensorBoard

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from IPython import display
from matplotlib import pyplot as plt

import numpy as np
import pathlib
import shutil
import tempfile


logdir = pathlib.Path(tempfile.mktemp()) / "tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
FEATURES = 28
# reading csv files straight from the gzip file without decompression
ds = tf.data.experimental.CsvDataset(gz, [float(), ] * (FEATURES + 1), compression_type="GZIP")


# this function reads the csv file
# returns a feature and a label for each record
def pack_row(*row):
    label = row[0]
    feature = tf.stack(row[1:], 1)
    return feature, label


# packing all the records into a data set
packed_ds = ds.batch(10000).map(pack_row).unbatch()  # combines all the elements into one array

for features, label in packed_ds.batch(1000).take(1):
    print(features[0])
    plt.hist(features.numpy().flatten(), bins=101)

n_validation = int(1000)  # validation set = 1000 samples
n_train = int(10000)  # training set = 10000 samples
buffer_size = int(1000)  # buffer size = 1000
batch_size = 500  # batch size = 500
steps_per_epoch = n_train // batch_size  # steps = 20 steps

# take(): takes 1000 elements from the packed_ds
# skip(): skips the first 1000 elements from the packed_ds
# cache(): makes sure the data isn't reloaded every epoch
validate_ds = packed_ds.take(n_validation).cache()  # creating the validation set
train_ds = packed_ds.skip(n_validation).take(n_train).cache()  # creating the training set
print(train_ds)

# preparing the data for training
validate_ds = validate_ds.batch(batch_size)  # creating batches of 500
# creating shuffled taining batches of 500
train_ds = train_ds.shuffle(buffer_size).repeat().batch(batch_size)

# reduces the learning rate of the model over time to prevent overfitting
# InverseTimeDecay(): hyperbolically decreases learning rate
# 1/2 at 1000 epochs, 1/3 at 2000 epochs ...
learn_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=steps_per_epoch * 1000,
    decay_rate=1,
    staircase=False)


def get_optimizer():
    return tf.keras.optimizers.Adam(learn_schedule)


step = np.linspace(0, 100000)
lr = learn_schedule(step)
plt.figure(figsize=(8, 600))
plt.plot(step / steps_per_epoch, lr)
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')


def get_callbacks(name):
    return [
        tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
        tf.keras.callbacks.TensorBoard(logdir/name)]


def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer = get_optimizer()  # adds the optimizer we made earlier
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  # loss equation is binary cross entropy
                  metrics=[
                      tf.keras.losses.BinaryCrossentropy(
                          from_logits=True, name='binary_crossentropy'), 'accuracy'])
    model.summary()
    # creating the model by plugging in the data
    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=max_epochs,
        validation_data=validate_ds,
        callbacks=get_callbacks(name),
        verbose=0)
    return history


size_histories = {}

# training a small model
# exponential linear unit activation
# 16 nodes in first layer
# 1 node in the last layer
tiny_model = tf.keras.Sequential([
    layers.Dense(16, activation='elu', input_shape=(FEATURES, )),
    layers.Dense(1)])

size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')


# creating a medium model
# 2 hidden layers with 16 units
small_model = tf.keras.Sequential([
    layers.Dense(16, activation='elu', input_shape=(FEATURES, )),
    layers.Dense(16, activation='elu'),
    layers.Dense(1)])

size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')


# creating a model for a medium dataset
# 3 hidden layers with 64 units each
medium_model = tf.keras.Sequential([
    layers.Dense(64, activation='elu', input_shape=(FEATURES, )),
    layers.Dense(64, activation='elu'),
    layers.Dense(64, activation='elu'),
    layers.Dense(1)])

size_histories['Medium'] = compile_and_fit(small_model, 'sizes/Medium')


# creating a large model
large_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES, )),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(1)])

size_histories['Large'] = compile_and_fit(large_model, 'sizes/Large')

plotter = tfdocs.plots.HistoryPlotter(metric='binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel("Epochs [log]")




