import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

tfds.disable_progress_bar()  # the training progress bar will not be shown
import numpy as np

print(tf.__version__)

(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',  # pre-encoded with ~8k vocabulary
    split=(tfds.Split.TRAIN, tfds.Split.TEST),  # returns the train and test data sets as a
    as_supervised=True,  # returns the data as (examples, labels)
    with_info=True)  # returns the 'info' structure with the data set

encoder = info.features['text'].encoder  # setting up the encoder 'SubwordTextEncoder'
print('Vocabulary size: ', encoder.vocab_size)  # prints out the size of the vocabulary

sample_string = 'Hello TensorFlow.'  # making a 'sample_string' variable

encoded_string = encoder.encode(sample_string)  # reversibly encodes any string passed into the encoder
print('Encoded string is ', encoded_string)

original_string = encoder.decode(encoded_string)  # decodes the string passed into the encoder
print('Original string is ', original_string)

for i in encoded_string:
    print(i, ' ----> ', encoder.decode([i]))

for train_example, train_label in train_data.take(1):  # [0] is not supported in tuples, we use .take() instead
    print('Encoded text: ', train_example[:10].numpy())  # removes tensor, shape and data type labels from the output
    print('Label: ', train_label.numpy())  # outputs the label of the first sentence of the data set

print(encoder.decode(train_example))

buffer_size = 1000  # setting the buffer size for the shuffler
train_batches = (
        train_data
        .shuffle(buffer_size)  # shuffles the train_data using the buffer_size set above
        .padded_batch(32))  # the reviews are different lengths, so we 0 pad each to batch them
test_batches = (test_data.padded_batch(32))

# each batch will have batch_size and a sequence_length because the padding is dynamic
# each batch will have a different length
for example_batch, label_batch in train_batches.take(2):
    print('Batch shape: ', example_batch.shape)
    print('Label shape: ', label_batch.shape)

model = keras.Sequential([
    keras.layers.Embedding(encoder.vocab_size, 16),
    # this is the embedding layer with 16 layers
    # where the embedding vector is looked up
    keras.layers.GlobalAveragePooling1D(),
    # this layer outputs a uniform vector regardless of the input (they are different)
    keras.layers.Dense(1)])
# there is a single output and a sigmoid activation

model.summary()

model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])  # using the Binary Cross Entropy loss equation because its a matter of 0s and 1s.

history = model.fit(train_batches,
                    epochs=10,
                    validation_data=test_batches,
                    validation_steps=30)

loss, accuracy = model.evaluate(test_batches)
print('Loss: ', loss)
print('Accuracy: ', accuracy)

history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.plot(epochs, acc, 'ro', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation (loss and accuracy)')
plt.xlabel('Epochs')
plt.ylabel('Loss (blue) & Accuracy (red)')
plt.legend()

plt.show()
