import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
import tensorflow_datasets as tfds

# downloading the data set from google's database
train_data, validation_data = tfds.load(
    name="cats_vs_dogs",
    split=('train[:80%]', 'train[80%:]'),  # dividing the data (80% training, 20% validation, no testing needed)
    as_supervised=True)  # downloads with the labels


# all our images will be reformatted to 160x160
img_dims = 160


# function to apply this augmentation to all elements of the data set
def format_example(image, label):
    image = tf.cast(image, tf.float32)  # makes sure that all the data are the same data type
    # normalising the data using a common function
    # all data is floats from 0 to 255
    # dividing by 127.5 brings it between 0 and 2
    # subtracting by 1 brings it between 0 and 1
    # easier for the model to process data between 0 and 1
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (img_dims, img_dims))  # resizing the data to 160x160
    return image, label


train = train_data.map(format_example)  # changing every training element
validation = validation_data.map(format_example)  # changing every validation element

batch_size = 32
buffer_size = 1000

train_batches = train.shuffle(buffer_size).batch(batch_size)  # creating the train batches
validation_batches = validation.batch(batch_size)  # creating the validation batches

# creating the model
model = tf.keras.Sequential([
    Conv2D(kernel_size=3,
           filters=16,
           padding='same',
           activation='relu',
           input_shape=[img_dims, img_dims, 3]),
    Conv2D(kernel_size=3,
           filters=30,
           padding='same',
           activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(kernel_size=3,
           filters=60,
           padding='same',
           activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(kernel_size=3,
           filters=90,
           padding='same',
           activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(kernel_size=3,
           filters=110,
           padding='same',
           activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(kernel_size=3,
           filters=130,
           padding='same',
           activation='relu'),
    Conv2D(kernel_size=1,
           filters=40,
           padding='same',
           activation='relu'),
    GlobalAveragePooling2D(),
    Dense(1, 'sigmoid')])


# compiling the model
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# training the model
history = model.fit(train_batches,
                    epochs=5,
                    validation_data=validation_batches)

# saving the model
model.save('cat_and_dog_model.h5')
