import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 28*28)/255
test_images = test_images[:1000].reshape(-1, 28*28)/255


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


# creating an initial model
model = create_model()
model.summary()

# creating a checkpoint path
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_images,
          train_labels,
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])

# seeing the accuracy of the untrained model
model = create_model()
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Untrained model, accuracy: ", (acc*100))

# loading the weights on the untrained model and then seeing the accuracy
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: ", (acc*100))


checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5)

# Create a new model instance
model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
model.fit(train_images,
          train_labels,
          epochs=50,
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
print(latest_checkpoint)

# making the new model and seeing the saved accuracy
model = create_model()
model.load_weights(latest_checkpoint)
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: ", (acc*100))


# saving the whole model using the HDF5 model save feature on keras
model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.save('my_model.h5')

# reloading that model
new_model = tf.keras.models.load_model('my_model.h5')
new_model.summary()

# checking the new accuracy
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: ", (acc*100))
