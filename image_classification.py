import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # this is to disable intel maximisation warnings (theyre a tad annoying)
# these are the packages I have used in this program
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# importing the dataset from the google database
fashion_mnist = keras.datasets.fashion_mnist
# this assigns these variables to the loaded arrays
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# giving names to the 10 types of garments
class_names = ['T-shirt/ top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape) # prints out the format of the array (60,000 images which are 28x28 pixels)
print(len(train_labels)) # prints the length of train_labels array
print(train_labels) # prints the whole train_labels array (the label given to each image)

print(test_images.shape)  # prints out the format of the array (10,000 images which are 28x28 pixels)
print(len(test_labels)) # prints the length of the test_labels array

plt.figure() # creates a matplotlib figure object
plt.imshow(train_images[0]) # shows the first image in the train_images array
plt.colorbar() # a coloring format of matplotlib colorbar library
plt.grid(False) # hides the grids of the figure object
plt.show() # shows the figure object

# normalises the values of the pixels (0-255 because it is b&w), we like numbers between 0 and 1
train_images = train_images / 255
test_images = test_images / 255

plt.figure(figsize=(10, 10)) # creates a custom 10x10 (inch) figure object
for i in range(10): # outputs the first 10 images in the test_images array
    plt.subplot(5, 5, i + 1) # creates multiple images in the same figure object
    plt.xticks([]) # marks the x-axis numbers
    plt.yticks([]) # marks the y-axis numbers
    plt.grid(False) # hides the grid
    plt.imshow(train_images[i], cmap=plt.cm.binary) # shows the image
    plt.xlabel(class_names[train_labels[i]]) # labels the x-axis with the label for the picture
plt.show()

# creates a model layer-by-layer (sequential)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # transforms 2D array into 1D array (784 elements)
    keras.layers.Dense(128, activation='relu'), # this NN layer has 128 nodes, this applies Rectified Linear Unit to each node
    keras.layers.Dense(10) # returns a logits array (probability function) with a length of 10 (the name classifiers)
])

# helps instantiate a model using specific arguments to increase accuracy
model.compile(optimizer='adam', # adam is an optimisation algorithm based on "Stochastic gradient descent"
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # measures the accuracy of the model during running
              # the model is optimized to make it go in the right direction
              metrics=['accuracy'] # monitors the training and testing steps, this uses the accuracy (the fraction of images correctly identified)
              )

# Training the model
# train_images and train_labels are fed into the model
# the model learns to associate the images and labels
# 10 epochs means that it will work through the set 10 times to get better
# verbose is the output format which keras uses to display learning progress

model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest Accuracy: ', test_acc) # outputs the test accuracy

# the softmax layer takes in the logits and outputs the probability of the relation to each classifier
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

print(predictions[0]) # outputs an array of probabilities relating the image to each classifier
print(np.argmax(predictions[0])) # uses the argmax to find the highest value in the array (the classifier is what the model predicts the image to be)


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i] # assigns values and arrays to local variables
    plt.grid(False) # the grid isn't displayed in the output
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary) # makes the picture black and white
    predicted_label = np.argmax(predictions_array) # the predicted label is the highest probability weighting in the classifiers
    if predicted_label == true_label:
        color = 'blue' # if correct, font is blue
    else:
        color = 'red' # if wrong, font is red
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)
    # label of the image: the prediction's name and accuracy is compared to the test_label and a suitable output is given.


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i] # assigns the suitable variables to arrays
    plt.grid(False)
    plt.xticks(range(10)) # 10 ticks on the x-axis
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777') # default bar colour is set to grey (777777)
    plt.ylim([0,1]) # sets the limit to 1 on the y-axis
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red') # if the prediction is wrong, the tallest bar is red
    thisplot[true_label].set_color('blue') # if the prediction is right, the tallest bar is blue


num_rows = 5 # the figure object will have 5 rows of images
num_columns = 3 # the figure object will have 3 columns of images
num_images = num_rows*num_columns # total number of images on the figure object
plt.figure(figsize=(2*2*num_columns, 2*num_rows)) # creates the figure object by taking in the arguments

# 2*2*num_columns because another plot will be made beside MNIST image
for i in range(num_images):
    plt.subplot(num_rows, 2*num_columns, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_columns, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout() # makes the plot nicely fit on the figure object
plt.show()
