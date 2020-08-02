import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print('Version: ', tf.__version__)  # output the version of tensorflow used
print('Eager mode: ', tf.executing_eagerly())  # evaluates answer immediately without building graphs
print('Hub version: ', hub.__version__)  # output the version of tensorflow hub used
print('GPU is',
      'available' if tf.config.experimental.list_physical_devices('GPU') else 'not available')  # outputs if GPU is used

train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),  # splits training set into 60% and 40% (60% for training, 40% for validation)
    # training sets: set used to calibrate the weights of the classifiers (self-learning)
    # validation sets: these are the sets used to fine tune weights set by the training sets to get a higher accuracy
    # test sets: set used to calculate the performance of the weights set by the training and validation sets (unbiased)
    as_supervised=True)  # returns a dataset with 2 distinct components (input and label)

# each element is one sentence
# each element also has a label
# 0: bad review
# 1: good review
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
# iter() makes an iteration object in the RAM, in this case it has 10 elements from train_data batch
# next just runs through the iter object 1 by 1.
print(train_examples_batch)  # prints out the elements in the examples batch
print(train_labels_batch)  # prints out the elements in the labels batch which shows if the element is positive or negative

# I am using a pre-trained text embedding model
# It is equally important to learn how to use premade models readily available online
# It would be quite difficult for a single person to make a sentiment analysis model

embedding = 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1'  # I am pretty much using the best possible embedded text saved model
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
# we plug the "embedding" which is the embedded text saved model into a layer
# where the input shape is the default shown by '[]'
# the data type of the input is string (obvs)
# and it is a trainable layer (which means the validation set can fine tune it)
print(hub_layer(train_examples_batch[:3]))
# this plugs the last 3 examples in the train_examples set
# the numbers in the array are how the sentence performs on the 50 dimensions provided by the model
# the specific operation of the dimension is beyond the scope of my learning right now

model = tf.keras.Sequential()  # creating a Sequential model
model.add(hub_layer)  # add the pre-made layer in the first input layer of the model
model.add(tf.keras.layers.Dense(16, activation='relu'))  # making the second layer with 16 nodes
model.add(tf.keras.layers.Dense(1))  # the third layer only has 1 node as its a binary answer (0 or 1)

print(model.summary())

model.compile(optimizer='adam',  # optimization algorithm used is 'adam'
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              # Binary cross entropy is the loss function used to computer losses in binary situations (0 or 1)
              metrics=['accuracy'])  # accuracy is the metric used to train the model

history = model.fit(train_data.shuffle(10000).batch(512),  # shuffles the tensor in buffer sizes of 10000, and then trains it in batches of 512
                    epochs=20,  # 20 iterations of training
                    validation_data=validation_data.batch(512),  # the validation data is also used to train in batches of 512
                    verbose=1)

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
