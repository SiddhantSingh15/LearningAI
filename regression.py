import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# downloading the data set
dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)  # shows the path to where the dataset has been saved to

# creates a table in pandas
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
print('\n')
print(
    dataset.tail().to_string())  # prints out the pandas table (tail() outputs the last 5, to_string() displays all columns)

print('\n')
print(dataset.isna().sum())  # finding the unknown values
dataset = dataset.dropna()
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})  # assigning names to the origin vals
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
print('\n')
print(dataset.tail().to_string())

train_dataset = dataset.sample(frac=0.8, random_state=0)  # uses 80% of the set as training set
test_dataset = dataset.drop(train_dataset.index)  # uses the rest (20%) by dropping the train_dataset

sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
plt.show()  # displays nice figures about the data

train_stats = train_dataset.describe()  # describe() uses stats tools to analyse the numbers
train_stats.pop('MPG')  # removing MPG (just playing around)
train_stats = train_stats.transpose()  # columns become rows and rows become columns
print('\n')
print(train_stats.to_string())  # printing the table

# I am removing this label because this is what we are trying to predict
train_labels = train_dataset.pop("MPG")  # removing the MPG label from the training dataset
test_labels = test_dataset.pop("MPG")  # removing the MPG label from the testing dataset


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


# normalising the data to prevent overfitting
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
print('\n')
print(normed_test_data.to_string())


def build_model():
    # the model has two densely connected layers which are hidden
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)])  # it will output a single continuous value for the MPG
    optimizer = tf.keras.optimizers.RMSprop(0.001)  # using the root mean squared optimiser

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])  # mae: mean absolute error, mse: mean square error
    return model


model = build_model()

model.summary()

example_batch = normed_train_data[:10]  # making a training batch
example_result = model.predict(example_batch)  # testing out the training batch on the model
print('\n')
print(example_result)

epochs = 1000  # setting the number of iterations on the model
#  storing the training and validation accuracy in the history object
history = model.fit(
    normed_train_data, train_labels,
    epochs=epochs, validation_split=0.2, verbose=0,
    callbacks=[tfdocs.modeling.EpochDots()])  # EpochDots() puts dots after each epoch for clarity


hist = pd.DataFrame(history.history)  # putting the train and val accuracy as well as losses in a table
hist['epoch'] = history.epoch  # labelling the epoch column
print('\n')
print(hist.tail().to_string())  # printing the hist table

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)  # plots the vals for every 2 iteration
plotter.plot({'Basic': history}, metric='mae')  # plots the mae metric from the history table
plt.ylim([0, 10])  # limits of y-axis
plt.ylabel('MAE [MPG]')  # label of the y-axis
plt.show()  # printing the figure object

model = build_model()

# stopping the training when the validation loss doesnt improve, checking every 10 epochs
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

print('\n')
# recompiling the model but stopping when early_stop is met
early_history = model.fit(normed_train_data, train_labels,
                          epochs=epochs, validation_split=0.2, verbose=0,
                          callbacks=[early_stop, tfdocs.modeling.EpochDots()])

plotter.plot({'Early Stopping': early_history}, metric='mae')  # this time monitoring validation loss
plt.ylim([0, 10])  # limits of the y-axis
plt.ylabel('MAE [MPG]')  # label of the y-axis
plt.show()  # printing the figure object

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)  # .evaluate() returns the average loss
print('\n')
print('Testing  MAE Error: ', mae, 'MPG')

test_predictions = model.predict(normed_test_data).flatten()
# seeing the variance in the predictions compared to the actual test data
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
limits = [0, 50]
plt.ylim(limits)
plt.xlim(limits)
plt.plot(limits, limits)
plt.show()

# printing out the error in the training
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
plt.ylabel('Count')
plt.show()
